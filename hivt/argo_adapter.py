"""
Adapter to interace with av1 dataset using av1 api.
"""
import copy
from pathlib import Path
from typing import Dict, Union, Tuple, List

import numpy as np
import torch
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.lane_segment import LaneSegment
from argoverse.map_representation.map_api import ArgoverseMap
from omegaconf import DictConfig

from trajpred._internal.utils import ConstraintApproximator as CA, RtsSmoother, Smoother
from trajpred.data_handling.base_dataset_adapter import BaseDatasetAdapter, DatasetName
from trajpred.data_handling.dataset_adapters.dataset_utils import DatasetUtils


class ArgoAdapter(BaseDatasetAdapter):

    def __init__(self, split_option: Union[str, Path], config: DictConfig, data_split: str = "train") -> None:
        super(ArgoAdapter, self).__init__(config, data_split=data_split)
        # FIXME Create a preprocess class
        if 'preprocessed' in self.config and self.config['preprocessed']:
            dataset = DatasetName.ARGO
            self.load_split(data_split, dataset)
        else:
            self.forecasting_loader = ArgoverseForecastingLoader(split_option)
            self.forecasting_loader.seq_list = sorted(self.forecasting_loader.seq_list)
            if self.config.data.num_splits_preprocess > 1:  # FIXME Put in utils
                n, N = self.config.data.split_preprocess_num, self.config.data.num_splits_preprocess
                sl = self.forecasting_loader.seq_list
                self.forecasting_loader.seq_list = sl[int(n / N * len(sl)): int((n + 1) / N * len(sl))]
                if len(self.forecasting_loader.seq_list) == 0:
                    raise ValueError(f"split_preprocess_num={n} is not meaningful. No samples to be processed.")
                print(f"Preprocessing split {n} of {N} with {len(self.forecasting_loader.seq_list)} samples.")

            self.map_loader = ArgoverseMap()
                

        self.rts_smoother = None
        polynomial_approximation_with_constraint = self.polynomial_approximation_with_constraint
        if self.config.data_features.approximate and self.config.data_features.approximate_with_rts:
            R, Q, P = self.config.preprocess_cfg.rts.R, self.config.preprocess_cfg.rts.Q, self.config.preprocess_cfg.rts.P
            self.rts_smoother = RtsSmoother(R, Q, P)

        self.smoother = Smoother(self.config, self.rts_smoother, polynomial_approximation_with_constraint)
        self.split_option = split_option
        print(f"Initialized {self.__class__.__name__} with {len(self)} samples.")

    def __len__(self):
        if 'preprocessed' in self.config and self.config['preprocessed']:
            return len(self.split)
        else:
            return len(self.forecasting_loader)

    def manipulate_data(self, data: Dict, gt: Dict) -> Dict:
        data['graph']['turn'] = torch.zeros_like(data['graph']['turn'])
        data['graph']['control'] = torch.zeros_like(data['graph']['control'])
        data['graph']['intersect'] = torch.zeros_like(data['graph']['intersect'])
        return data, gt

    def read_dataset_data(self, idx: int) -> Dict:
        city = copy.deepcopy(self.forecasting_loader[idx].city)
        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.forecasting_loader[idx].seq_df)

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        steps = [mapping[x] for x in df['TIMESTAMP'].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]
        
        #TODO: make sure it works with agent as first, av as second, then others
        # git dif -> compares 
        # add on readme 
        agt_idx = obj_type.index('AGENT')
        av_idx = obj_type.index('AV')
        
        idcs_agt = objs[keys[agt_idx]]
        idcs_av  = objs[keys[av_idx]]

        agt_traj = trajs[idcs_agt]
        agt_step = steps[idcs_agt]
        
        av_traj = trajs[idcs_av]
        av_step = steps[idcs_av]

        del keys[agt_idx]
        del keys[av_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] +[av_traj]+ ctx_trajs
        data['steps'] = [agt_step] +[av_step]+ ctx_steps

        # Interpolate
        if self.config.data_features.interpolate:
            self.interpolate_data(data, mapping)

        if self.config.data_features.approximate:  # FIXME Put that into the lanegcn get_obj_features function? -> Smoothing in local coord system might be more beneficial
            data = self.smoother.approximate_data(data)
        real_data_idx = self.forecasting_loader.seq_list[idx].stem
        print('ARGO_ADAPTER LINE 110, the idx, or seq_id is:',real_data_idx)
        data['idx'] = int(real_data_idx)
        return data

    def read_map_data(self, data: Dict) -> Dict:
        orig, rot, data = DatasetUtils.get_orig_rot_from_trajectory(self.config, data, self.split_option)
        if orig is None:
            return None
        lane_ids, lanes = self.get_lanes_in_specified_radius_and_transform(orig=orig, rot=rot, city=data['city'])
        data['map_img'] = None
        data['map_pnts'] = {'lane_ids': lane_ids, 'lanes': lanes}
        return data

    # Function for interpolation
    def interpolate_data(self, data: Dict, mapping: Dict) -> Dict:
        DELTA_T = 0.1
        for i_traj in range(len(data['trajs'])):
            traj_ = data['trajs'][i_traj]
            t_origin = (np.array(list(mapping.keys())) - list(mapping.keys())[0])[:traj_.shape[0]]
            t_interp = np.linspace(0, (traj_.shape[0] - 1) * DELTA_T, traj_.shape[
                0])  # np.arange(0,t_origin[-1]-t_origin[-1]%DELTA_T+DELTA_T, DELTA_T) #
            traj_interp = np.array(
                [np.interp(t_interp, t_origin, traj_[:, 0]), np.interp(t_interp, t_origin, traj_[:, 1])]).T
            data['trajs'][i_traj] = traj_interp

            # import matplotlib.pyplot as plt
            # plt.plot(t_origin, traj_[:,0], '.', label="raw")
            # plt.plot(t_interp, traj_interp[:,0], '.', label="interp")
            # plt.legend();
            # plt.show()
            # plt.plot(traj_[:, 0], traj_[:, 1]);
            # plt.plot(traj_interp[:, 0], traj_interp[:, 1]);
            # plt.show()

        return data

    # Functions for approximation and smoothing
    def get_future_and_past_data(self, t, traj):
        idx_begin_past = 0
        idx_begin_fut = self.config.data_features.ts_origin + 1
        sel_past = (t >= idx_begin_past) & (t < idx_begin_fut)
        sel_fut = t >= idx_begin_fut
        past = traj[sel_past]
        fut = traj[sel_fut]
        t_past = t[sel_past]
        t_fut = t[sel_fut]
        return t_past, past, t_fut, fut

    # TODO It would be best to remove the smoothing functions that are not used anymore (at least in the master branch).
    def polynomial_approximation_with_constraint(self, t, y, order=4):
        t_past, past, t_fut, fut = self.get_future_and_past_data(t, y)
        # print(t_past.shape, fut.shape)
        if t_past.shape[0] == 5:
            a = 1

        order_past, d_hat_past = CA.correct_order(t_past, y, order)
        order_fut, d_hat_fut = CA.correct_order(t_fut, y, order)

        if (order_past <= 0) and (order_fut <= 0):
            # print("WARNING: No approximation possible. Using raw data.")
            return t, y
        if order_past <= 0:
            # print("Warning: order_past <= 0")
            fut_hat = CA.approximate_polynomial_under_constraints(t_fut, fut, order=order_fut)
            hat = fut_hat
            if order_past == 0:
                hat = np.concatenate((past, fut_hat))
            return t, hat
        if order_fut <= 0:
            # print("Warning: order_fut <= 0")
            past_hat = CA.approximate_polynomial_under_constraints(t_past, past, order=order_past)
            hat = past_hat
            if order_fut == 0:
                hat = np.concatenate((past_hat, d_hat_fut))
            return t, hat

        # Approximate with two polynomials and continous derivatives
        order_fut = min(order_fut + 1, order)
        t_fut_ = np.concatenate(([t_past[-1]], t_fut))
        fut_ = np.concatenate(([past[-1]], fut))

        # Get system matrix A
        # num_param = 2 * (order + 1)
        num_param = order_past + order_fut + 2
        A_past = np.array([np.power(ti, range(order_past + 1)) for ti in t_past])
        A_fut = np.array([np.power(ti, range(order_fut + 1)) for ti in t_fut_])
        A = np.zeros((len(t) + 1, num_param))
        A[:len(t_past), : order_past + 1] = A_past
        A[len(t_past):, order_past + 1:] = A_fut

        # Get Lagrange (constraint) matrix C
        num_constraints = 4  # +1 if the value needs to be fixed
        C = np.zeros((num_constraints, num_param))

        # Boundary conditions d
        # d = np.array([yi,yi,0]) #If the value needs to be fixed
        # d = np.array([0,0])
        d = np.array([0, past[0], fut[-1], 0])

        c_pos_past_trans = np.zeros(num_param)
        c_pos_past_trans[:order_past + 1] = A_past[-1, :]
        C[0, :] = c_pos_past_trans

        c_pos_fut_trans = np.zeros(num_param)
        c_pos_fut_trans[order_past + 1:] = A_fut[0, :]
        # C[1,:] = c_pos_fut_trans #If the value needs to be fixed
        C[0, :] -= c_pos_fut_trans
        # End and start condition
        C[1, :] = A[0, :]
        C[2, :] = A[-1, :]

        ti = t_past[-1]

        c_1st_deriv_transition = np.zeros(num_param)
        c_1st_deriv_transition[1] = c_1st_deriv_transition[order_past + 1 + 1] = 1
        c_1st_deriv_transition[2:order_past + 1] = np.array([k * (ti ** (k - 1)) for k in range(2, order_past + 1)])
        c_1st_deriv_transition[order_past + 1 + 2:] = np.array(
            [k * (ti ** (k - 1)) for k in range(2, order_fut + 1)])  # c_1st_deriv_transition[2:num_param // 2]
        c_1st_deriv_transition[order_past + 1 + 1:] *= -1
        C[-1, :] = c_1st_deriv_transition

        b = np.concatenate((past, fut_))

        p = CA.solve_lse_and_approximate(A, b, C, d, num_constraints)

        ts_begin = self.config.data_features.ts_origin + 1
        t_approx = np.concatenate((t_past, t_fut_))
        # t_approx[np.where(t_approx == (ts_begin-1))[0][1]] = 20
        past_hat = p @ A[t_approx < ts_begin][:-1].T
        fut_hat = p @ A[t_approx >= ts_begin].T

        hat = np.concatenate((past_hat, fut_hat))
        return t, hat

    def approximate_with_two_polynomials_constrained(self, t, y, order=4, constrain_mode="both+derivative"):
        DELTA_T = 0.1  # FIXME Param
        t_past, past, t_fut, fut = self.get_future_and_past_data(t, y)
        if (constrain_mode == "transition") and (len(t_fut) < 1 or len(t_past) < 1):
            constrain_mode = "both"
            print("-> Using constrain_mode both")

        # Computation of derivative: mean of numerical derivative or approximate and take mean of the derivatives of the two polynoms

        def mean_gradient(t, y, past=True):
            mean_horizon = 5
            num_samples = min(len(t_past), mean_horizon)
            end = -1 if past else num_samples
            start = -num_samples if past else 1
            return np.mean(np.gradient(y, t)[start:end]) * DELTA_T

        def mean_2nd_deriv(t, grad, past=True):
            mean_horizon = 5
            num_samples = min(len(t_past), mean_horizon)
            end = -2 if past else num_samples
            start = -num_samples if past else 2
            return np.mean(np.gradient(grad, t)[start:end]) * DELTA_T

        if len(fut) == 0:
            y_transition = past[-1]
            der_transition = mean_gradient(t_past, past, past=True)
        elif len(past) == 0:
            y_transition = fut[0]
            der_transition = mean_gradient(t_fut, fut, past=False)
        else:
            y_transition = 1 / 2 * (past[-1] + fut[0]) if len(fut) > 0 else past[-1]
            der_transition = 1 / 2 * (mean_gradient(t_fut, fut, past=False) + mean_gradient(t_past, past, past=True))
            # FIXME Do it or all options
            der2_transition = 1 / 2 * (
                    mean_2nd_deriv(t_fut, np.gradient(fut, t_fut), past=False) + mean_2nd_deriv(t_past,
                                                                                                np.gradient(past,
                                                                                                            t_past),
                                                                                                past=True))
            der_transition = (der_transition, der2_transition)
        try:
            past_hat, p = CA.approximate_one_polynomial_under_constraint(t_past, past, order=order,
                                                                         constrain_mode=constrain_mode,
                                                                         der_transition=der_transition,
                                                                         y_transition=y_transition, t_mode="past")
            # x = np.array([t_fut[0] ** i for i in range(order + 1)]).T
            # y_transition = p @ x.T #FIXME
            fut_hat, _ = CA.approximate_one_polynomial_under_constraint(t_fut, fut, order=order,
                                                                        constrain_mode=constrain_mode,
                                                                        der_transition=der_transition,
                                                                        y_transition=y_transition,
                                                                        t_mode="fut")
        except:
            print("Error when solving approximation problem. Do not use the 1st derivative constraint.")
            constrain_mode = "both"
            past_hat, p = CA.approximate_one_polynomial_under_constraint(t_past, past, order=order,
                                                                         constrain_mode=constrain_mode,
                                                                         der_transition=der_transition,
                                                                         y_transition=y_transition,
                                                                         t_mode="past")
            fut_hat, _ = CA.approximate_one_polynomial_under_constraint(t_fut, fut, order=order,
                                                                        constrain_mode=constrain_mode,
                                                                        der_transition=der_transition,
                                                                        y_transition=y_transition,
                                                                        t_mode="fut")
        # print(fut_hat[0] - past_hat[-1])
        y_hat = np.hstack((past_hat, fut_hat))
        t_hat = np.hstack((t_past, t_fut))
        return t_hat, y_hat

    def approximate_traj_with_polynomial(self, t, y, order=4):
        px = np.poly1d(np.polyfit(t, y, order))
        y_hat = px(t)
        return y_hat

    def get_lanes_in_specified_radius_and_transform(self, orig, rot, city) -> Tuple[List[int], Dict[int, LaneSegment]]:
        radius = DatasetUtils.get_radius_from_pred_range(self.config.data_loader.pred_range)

        lane_ids = self.map_loader.get_lane_ids_in_xy_bbox(
            orig[0], orig[1], city, radius)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()

        for lane_id in lane_ids:
            lane = self.map_loader.city_lane_centerlines_dict[city][lane_id]
            lane = copy.deepcopy(lane)
            # polygon is not needed here
            # polygon = self.map_loader.get_lane_segment_polygon(
            #             lane_id, city)
            # lane.polygon = polygon[:, :2]
            lanes[lane_id] = lane
        lane_ids = list(lanes.keys())

        # for lane_id in lane_ids:
        #     lane = self.map_loader.city_lane_centerlines_dict[city][lane_id]
        #     lane = copy.deepcopy(lane)
        #     # Transform into agent frame
        #     centerline = np.matmul(
        #         rot, (lane.centerline - orig.reshape(-1, 2)).T).T
        #     x, y = centerline[:, 0], centerline[:, 1]
        #     if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
        #         continue
        #
        #     """Getting polygons requires original centerline"""
        #     polygon = self.map_loader.get_lane_segment_polygon(
        #         lane_id, city)
        #     polygon = copy.deepcopy(polygon)
        #     lane.centerline = centerline
        #     lane.polygon = np.matmul(
        #         rot, (polygon[:, :2] - orig.reshape(-1, 2)).T).T
        #     lanes[lane_id] = lane
        # lane_ids = list(lanes.keys())
        return lane_ids, lanes
