import copy
import pickle
import os
import random
import time
import sys

import numpy as np

from .utils import get_node_matrix, is_begin, mydeepcopy, NO_DEALY_T
from .utils import ExchangeOperator, TSOperator

from src.common.node import Node
from src.common.route import Map
from src.conf.configs import Configs
from src.utils.input_utils import get_factory_info, get_route_map
from src.utils.json_tools import convert_nodes_to_json
from src.utils.json_tools import get_vehicle_instance_dict, get_order_item_dict
from src.utils.json_tools import read_json_from_file, write_json_to_file
from src.utils.logging_engine import logger

node_matrix, node_int2str, node_str2int = get_node_matrix()
DOCK_TIME = Configs.DOCK_APPROACHING_TIME
LOAD_TIME_UNIT = int(1 / Configs.LOAD_SPEED * 60)  # 单位demand 装货/卸货用时：240s
MAX_CAPACITY = 15
save_path = os.path.join(Configs.algorithm_folder_path, 'order_dict.pkl')
planed_route_path = os.path.join(Configs.algorithm_folder_path, 'planed_route.pkl')
info_path = os.path.join(Configs.algorithm_folder_path, 'info.pkl')


def get_vehicle_space(vehicle):
    """
    获取车辆当前剩余容量
    :param vehicle: 车辆对象实例
    :return: space 剩余空间
    """
    tmp = copy.deepcopy(vehicle.carrying_items)
    space = vehicle.board_capacity
    while not tmp.is_empty():
        space -= tmp.pop().demand
    return space


def calculate_demand(item_list: list):
    demand = 0
    for item in item_list:
        demand += item.demand
    return demand


class Solution(object):
    """
    路径解类：包含每辆车的规划节点路径顺序
    """
    def __init__(self, vehicle_id_to_planned_route):
        self.vehicle_id_to_planned_route = vehicle_id_to_planned_route
        self.vehicle_id_to_assigned_order = {}
        self.time_window = {}  # 每辆车路径每个节点的时间窗
        self.space = {}  # 每辆车在每个节点处的容量
        self.dis = {}  # 每辆车的路径长度
        self.over_time = {}  # 该解的预估超时时间
        self.order_over_time = {}
        # self.cost = {}  # 每条路径的cost
        self.total_dis = 0  # 该解的预估总路程
        self.total_over_time = 0
        self.total_cost = 9999999999  # 代价 = 总超时小时数*10000 + 车辆平均km (时间秒距离/120)
        self.mask = {}  # 首节点能否改变，0表示能，1表示不能
        self.k = len(vehicle_id_to_planned_route)
        self.split_order_set = set([])
        self.pickup_index = {}
        self.last_pos = {}  # 路径的最后位置

    def init_time_window(self, id_to_vehicle, vid_list=None):
        """
        计算各个路径点的[arrive, leave]时间窗, 剩余容量
        :return:
        """
        if vid_list is None:  # 全新的解
            self.order_over_time = {}
            self.split_order_set = set([])
        for vid, route in self.vehicle_id_to_planned_route.items():
            if vid_list is not None and vid not in vid_list:  # 不在待更新列表中就跳过
                continue
            self.time_window[vid] = []
            self.space[vid] = []
            self.mask[vid] = 0
            self.pickup_index[vid] = []
            self.last_pos[vid] = -1
            FIRST = True
            for idx in range(len(route)):
                if FIRST:
                    # 首个节点两种情况，1）首节点已经是当前destination了 2）车子当前为停车状态，节点是新规划的还没传入模拟器
                    # destination已经固定，直接取车辆信息中的到达时间
                    if id_to_vehicle[vid].destination is not None:
                        arrive_time = id_to_vehicle[vid].destination.arrive_time
                        leave_time = arrive_time + DOCK_TIME + route[idx].service_time
                        # leave_time_d = id_to_vehicle[vid].destination.leave_time
                        self.mask[vid] = 1  # destination固定了，不能更改
                    else:  # 新规划的，需要自己算时间
                        v = id_to_vehicle[vid]  # Vehicle object
                        now = node_str2int[v.cur_factory_id]
                        next = route[idx].int_id
                        arrive_time = v.leave_time_at_current_factory + node_matrix[now][next]
                        leave_time = arrive_time + DOCK_TIME + route[idx].service_time
                        if route[idx].order_index > 0:  # 存储取货点idx
                            self.pickup_index[vid].append(idx)
                    # 计算初始剩余容量
                    space = get_vehicle_space(id_to_vehicle[vid])
                    FIRST = False
                else:
                    now = route[idx-1].int_id
                    next = route[idx].int_id
                    if now == next:
                        leave_time = leave_time + route[idx].service_time
                    else:
                        arrive_time = leave_time + node_matrix[now][next]
                        leave_time = arrive_time + DOCK_TIME + route[idx].service_time
                    if route[idx].order_index > 0:  # 存储取货点idx
                        self.pickup_index[vid].append(idx)
                # 计算剩余容量
                unload_demand = calculate_demand(route[idx].delivery_items)  # 卸载容量
                load_demand = calculate_demand(route[idx].pickup_items)
                space = space + unload_demand - load_demand
                self.space[vid].append(space)
                self.time_window[vid].append([arrive_time, leave_time])

                if route[idx].order_index < -10000:
                    self.split_order_set.add(abs(route[idx].order_index) % 10000)
            if len(route) > 0:
                self.last_pos[vid] = route[-1].int_id
            else:
                last_factory_id = id_to_vehicle[vid].cur_factory_id
                self.last_pos[vid] = node_str2int[last_factory_id]

    def calc_dis(self, id_to_vehicle, vid_list=None):
        if id_to_vehicle is None:
            raise ValueError("车辆信息为空！")
        for vid, route in self.vehicle_id_to_planned_route.items():
            if vid_list is not None and vid not in vid_list:  # 不在待更新列表中就跳过
                continue
            v_dis = 0
            first = True
            for idx in range(len(route)):
                if first:
                    # 首个节点两种情况，1）首节点已经是当前destination了 2）车子当前为停车状态，节点是新规划的还没传入模拟器
                    # 已经固定为destination，对所有情况都一样，不考虑开过去的cost
                    if id_to_vehicle[vid].destination is not None:
                        first = False
                        continue
                    else:  # 新规划的，需要自己算路程
                        v = id_to_vehicle[vid]  # Vehicle object
                        now = node_str2int[v.cur_factory_id]
                        next = route[idx].int_id
                        v_dis += node_matrix[now][next]
                    first = False
                else:
                    last = route[idx-1].int_id
                    now = route[idx].int_id
                    v_dis += node_matrix[last][now]
            self.dis[vid] = v_dis
        self.total_dis = sum(self.dis.values())

    def calc_over_time(self, vid_list=None, update=False, id_to_vehicle=None):
        # 如果启用update参数，一定要跟 id_to_vehicle参数！
        # 重置拆分订单的超时
        for s in self.split_order_set:
            self.order_over_time[s] = 0
        if update:
            self.init_time_window(id_to_vehicle=id_to_vehicle, vid_list=vid_list)
        for vid, route in self.vehicle_id_to_planned_route.items():
            if vid_list is not None and vid not in vid_list:  # 不在待更新列表中，只考虑拆分订单
                for idx in range(len(route)):
                    order_id = abs(route[idx].order_index) % 10000
                    if order_id in self.split_order_set and route[idx].order_index < 0:  # 拆分节点
                        t_finish = self.time_window[vid][idx][1]
                        t_commit = route[idx].delivery_items[0].committed_completion_time
                        ot = max(0, t_finish - t_commit)
                        self.order_over_time[order_id] = max(self.order_over_time[order_id], ot)
                continue
            self.over_time[vid] = 0
            for idx in range(len(route)):
                if route[idx].order_index < 0:  # 送货节点
                    t_finish = self.time_window[vid][idx][1]
                    t_commit = route[idx].delivery_items[0].committed_completion_time
                    ot = max(0, t_finish - t_commit)
                    order_id = abs(route[idx].order_index) % 10000
                    if order_id in self.split_order_set and order_id in self.order_over_time:  # 是拆分订单，用最后完成的
                        self.order_over_time[order_id] = max(self.order_over_time[order_id], ot)
                    else:
                        self.order_over_time[order_id] = ot
                    if route[idx].merge:  # 合并订单，计算每个子订单的超时
                        for item_idx in range(1, len(route[idx].delivery_items)):
                            if route[idx].delivery_items[item_idx].committed_completion_time != t_commit:
                                t_commit = route[idx].delivery_items[item_idx].committed_completion_time
                                ot = max(0, t_finish - t_commit)
                                self.order_over_time[order_id] += ot
                    self.over_time[vid] += self.order_over_time[order_id]

        self.total_over_time = sum(self.order_over_time.values())

    def calc_cost(self, vid_list=None, update=False, id_to_vehicle=None):
        # 如果启用update参数，一定要跟 id_to_vehicle参数！
        if update:
            self.calc_dis(id_to_vehicle=id_to_vehicle, vid_list=vid_list)
            self.calc_over_time(vid_list=vid_list, update=update, id_to_vehicle=id_to_vehicle)

        self.total_cost = (self.total_over_time / 3600 * Configs.LAMDA) + (self.total_dis / 120 / self.k)

    def update_solution(self, target, vid_list=None):
        """
        允许只更新部分内容，降低copy操作的消耗
        """
        if vid_list is None:
            self.vehicle_id_to_planned_route = copy.deepcopy(target.vehicle_id_to_planned_route)
        else:
            for vid in vid_list:
                self.vehicle_id_to_planned_route[vid] = copy.deepcopy(target.vehicle_id_to_planned_route[vid])

        self.time_window = mydeepcopy(target.time_window)
        self.order_over_time = mydeepcopy(target.order_over_time)
        self.dis = mydeepcopy(target.dis)
        self.pickup_index = mydeepcopy(target.pickup_index)
        self.space = mydeepcopy(target.space)

        self.total_dis = target.total_dis
        self.total_over_time = target.total_over_time
        self.total_cost = target.total_cost


class MyOrder(object):
    def __init__(self, order_id: str, order_index: int, items: list, merge=False):
        self.order_id = order_id
        self.order_index = int(order_id[-4:])  # 从1开始
        self.items = items
        self.demand = sum([x.demand for x in items])  # 总需求
        self.load_time = sum([x.load_time for x in items])  # 装货时间
        self.unload_time = sum([x.unload_time for x in items])
        self.pickup_factory_id = items[0].pickup_factory_id  # 取货点
        self.delivery_factory_id = items[0].delivery_factory_id
        self.create_time = items[0].creation_time
        self.committed_completion_time = items[0].committed_completion_time
        self.time_window = [self.create_time, self.committed_completion_time]
        self.delivery_state = items[0].delivery_state  # 0：新生成（和模拟器定义不同）1: 已生成(generated), 2: 进行中(ongoing), 3: 完成(Completed)

        self.merge = merge
        self.split = 0  # 被拆成了几份


class MyNode(Node):
    def __init__(self, factory_id: str, lng: float, lat: float, pickup_item_list: list, delivery_item_list: list):
        super().__init__(factory_id, lng, lat, pickup_item_list, delivery_item_list)
        self.order_index = 0  # N代表第N个订单取货节点，-N代表第N个订单送货节点
        self.int_id = node_str2int[factory_id]
        self.merge = False

    def set_order_index(self, index):
        if len(self.pickup_items) > 0:
            self.order_index = index
        else:
            self.order_index = -1 * index


def first_solution(id_to_unallocated_order_item: dict, id_to_vehicle: dict, id_to_factory: dict):
    t_start = time.time()
    vehicle_id_to_planned_route = {}  # 存放每辆车的路径规划（MyNode list）

    for vehicle_id in id_to_vehicle:
        MAX_CAPACITY = id_to_vehicle[vehicle_id].board_capacity
        break

    id_to_all_myorder = {}  # 存储已注册分配历史订单
    last_planed_route = {}  # 存储路径
    info = {}  # 存储信息
    delay_cnt = 0  # 过去已经连续延迟了几次
    interval = 1
    if not is_begin(id_to_vehicle):
        if os.path.exists(save_path):  # 要考虑下怎么处理不同case切换时的情况
            id_to_all_myorder = pickle.load(open(save_path, 'rb'))
        if os.path.exists(planed_route_path):
            last_planed_route = pickle.load(open(planed_route_path, 'rb'))
        if os.path.exists(info_path):
            info = pickle.load(open(info_path, 'rb'))
            delay_cnt = info['delay_cnt']
            interval = info['interval']

    # 初始化规划路径
    for vehicle_id in id_to_vehicle:
        vehicle_id_to_planned_route[vehicle_id] = []
    # 处理旧路径中已经达到的部分
    for vehicle_id, route in last_planed_route.items():
        if id_to_vehicle[vehicle_id].destination is None:
            continue
        else:
            destination_id = id_to_vehicle[vehicle_id].destination.id
            idx = 0
            while idx < len(route):
                if route[idx].id == destination_id:
                    break
                idx += 1
            vehicle_id_to_planned_route[vehicle_id] = route[idx:]

    delay = False
    if len(id_to_vehicle) > 10 and interval > 15 and interval not in NO_DEALY_T:  # 超过10辆车才考虑延迟分配
        if delay_cnt > 0:
            if random.random() < 0.5 - delay_cnt * 0.2:
                delay = True
            else:
                delay = False
        else:
            delay = True

    order_id_to_items = {}
    interval_list = [12, 42, 78, 110, 145]  # 进行完全重分配的时间窗序号
    # 重分配所有
    if interval in interval_list:
        order_id_to_items, vehicle_id_to_planned_route = re_assign(id_to_unallocated_order_item,
                                                id_to_vehicle, id_to_factory, id_to_all_myorder)
        print('*** 重分配 *** interval =', interval)
        # 重新注册
        for order_id, item_list in order_id_to_items.items():
            ids = []
            # if order_id not in id_to_all_myorder:  # 新订单货物
            for item in item_list:
                if item.order_id not in ids:
                    ids.append(item.order_id)
            for i in ids:
                idx = len(id_to_all_myorder) + 1
                if len(ids) > 1:
                    id_to_all_myorder[i] = MyOrder(i, idx, item_list, merge=True)
                else:
                    id_to_all_myorder[i] = MyOrder(i, idx, item_list)
    else:  # 只分配新订单
        # 构造 {新订单id：货物列表} 字典 (只保存新订单)
        for item_id, item in id_to_unallocated_order_item.items():  # 遍历每个货物
            # if item_id in pre_matching_item_ids:  # 已经分配，车子在路上了， 不能再分配了 (如果是合单运送，也可以重分配但没必要)
            #     continue
            order_id = item.order_id
            if order_id in id_to_all_myorder:  # 已经注册，说明分配了
                continue
            if order_id not in order_id_to_items:
                order_id_to_items[order_id] = []
            order_id_to_items[order_id].append(item)

        if not delay:  # 没有延迟，合并后分配所有订单
            delay_cnt = 0
            # 注册即将分配的订单
            order_id_to_items = merge_order(order_id_to_items)  # 合并订单
            for order_id, item_list in order_id_to_items.items():
                ids = []
                # if order_id not in id_to_all_myorder:  # 新订单货物
                for item in item_list:
                    if item.order_id not in ids and item.order_id not in id_to_all_myorder:
                        ids.append(item.order_id)
                for i in ids:
                    idx = len(id_to_all_myorder) + 1
                    if len(ids) > 1:
                        id_to_all_myorder[i] = MyOrder(i, idx, item_list, merge=True)
                    else:
                        id_to_all_myorder[i] = MyOrder(i, idx, item_list)
        else:  # 延迟分配，合并后只先分配大订单
            delay_cnt += 1
            print('延迟分配', delay_cnt)
            order_id_to_items = merge_order(order_id_to_items)  # 合并订单
            tmp_order_id_to_items = {}
            for order_id, item_list in order_id_to_items.items():
                ids = []
                if calculate_demand(item_list) > MAX_CAPACITY-1:
                    tmp_order_id_to_items[order_id] = item_list
                    ids.append(order_id)
                    for item in item_list:
                        if item.order_id not in ids:
                            ids.append(item.order_id)
                    for i in ids:
                        idx = len(id_to_all_myorder) + 1
                        if len(ids) > 1:
                            id_to_all_myorder[i] = MyOrder(i, idx, item_list, merge=True)
                        else:
                            id_to_all_myorder[i] = MyOrder(i, idx, item_list)
            order_id_to_items = tmp_order_id_to_items

    # ****************************** 订单分配部分 *********************
    solution = Solution(copy.deepcopy(vehicle_id_to_planned_route))  # 临时解
    solution.calc_cost(update=True, id_to_vehicle=id_to_vehicle)
    if len(order_id_to_items) > 0:
        # 现在订单都在 {id: MyOrder object} 字典里了
        # vehicles = [vehicle for vehicle in id_to_vehicle.values()]  # 车辆objects 列表
        for order_id, items in order_id_to_items.items():  # 依次查看决策每个订单
            # 超过最大容量 需要拆分
            if id_to_all_myorder[order_id].demand > MAX_CAPACITY:  # 超过最大容量，需要拆分
                cur_demand = 0  # demo策略只分配给空车
                tmp_items = []
                for item in items:
                    if cur_demand + item.demand > MAX_CAPACITY:  # 这辆车满了，分配清单
                        # 选定插入车辆，更新路径
                        id_to_all_myorder[order_id].split += 1
                        split = id_to_all_myorder[order_id].split
                        # 构造节点
                        pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(tmp_items, id_to_factory)
                        pickup_node.set_order_index(id_to_all_myorder[order_id].order_index + split * 100000)  # order index，可检查LIFO
                        delivery_node.set_order_index(id_to_all_myorder[order_id].order_index + split * 100000)
                        # 基于距离选择路径
                        vehicle_id = min(solution.dis, key=lambda k: solution.dis[k] + node_matrix[solution.last_pos[k]][pickup_node.int_id])
                        # vehicle = vehicles[vehicle_index]
                        solution.vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
                        solution.vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
                        # print('订单 {}-{} 插入 车辆 {}'.format(order_id, split, vehicle.id))
                        solution.calc_cost(vid_list=[vehicle_id], update=True, id_to_vehicle=id_to_vehicle)
                        tmp_items = []
                        cur_demand = 0

                    tmp_items.append(item)
                    cur_demand += item.demand

                if len(tmp_items) > 0:  # 处理最后轮到的一辆车
                    # 选定插入车辆，更新路径
                    id_to_all_myorder[order_id].split += 1
                    split = id_to_all_myorder[order_id].split
                    # 构造节点
                    pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(tmp_items, id_to_factory)
                    pickup_node.set_order_index(
                        id_to_all_myorder[order_id].order_index + split * 100000)  # order index，可检查LIFO
                    delivery_node.set_order_index(id_to_all_myorder[order_id].order_index + split * 100000)
                    # 选择长度最短的路径
                    vehicle_id = min(solution.dis, key=lambda k: solution.dis[k] + node_matrix[solution.last_pos[k]][pickup_node.int_id])
                    # vehicle = vehicles[vehicle_index]
                    solution.vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
                    solution.vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
                    # print('订单 {}-{} 插入 车辆 {}'.format(order_id, split, vehicle.id))
                    solution.calc_cost(vid_list=[vehicle_id], update=True, id_to_vehicle=id_to_vehicle)

            else:  # 不用拆分，基于规则选一辆车插到最后面
                pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(items, id_to_factory)  # 打包成Node
                pickup_node.set_order_index(id_to_all_myorder[order_id].order_index)  # 设置Node的 order index，用于检查LIFO
                delivery_node.set_order_index(id_to_all_myorder[order_id].order_index)
                if id_to_all_myorder[order_id].merge:
                    pickup_node.merge = delivery_node.merge = True

                vehicle_id = min(solution.dis, key=lambda k: solution.dis[k] + node_matrix[solution.last_pos[k]][pickup_node.int_id])
                # print('插入', vehicle_id)
                solution.vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
                solution.vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
                # print('订单 {} 插入 车辆 {}'.format(order_id, vehicle.id))
                solution.calc_cost(vid_list=[vehicle_id], update=True, id_to_vehicle=id_to_vehicle)

            # vehicle_index = (vehicle_index + 1) % len(vehicles)  # 换下一辆

    # print('时间窗：[{}]'.format(interval))
    interval += 1

    print('优化前已用时 {:2f} s'.format(time.time() - t_start), file=sys.stderr)
    t = 2
    times = 5
    if interval > 144:
        t = 3
    if interval in interval_list + [x+1 for x in interval_list]:
        times = 10
    time_used = time.time() - t_start
    w = [10, 10, 5, 5]
    vehicle_id_to_planned_route = optimize_solution(solution, id_to_vehicle, time_used, w, times, t)

    # info['vehicle_index'] = vehicle_index
    info['delay_cnt'] = delay_cnt
    info['interval'] = interval
    pickle.dump(id_to_all_myorder, open(save_path, 'wb'))  # 保存成pkl文件
    pickle.dump(vehicle_id_to_planned_route, open(planed_route_path, 'wb'))  # 保存成pkl文件
    # pickle.dump(vehicle_index, open(vehicle_index_path, 'wb'))  # 保存成pkl文件
    pickle.dump(info, open(info_path, 'wb'))

    # empty_list = []
    # for v, route in vehicle_id_to_planned_route.items():
    #     if len(route) == 0:
    #         empty_list.append(v)
    # print('空路径数量', len(empty_list))
    print('总用时 {:2f} s'.format(time.time() - t_start), file=sys.stderr)

    return vehicle_id_to_planned_route


def optimize_solution(solution, id_to_vehicle, time_used, w, times, t=2):
    opr = ExchangeOperator(solution, id_to_vehicle)
    # opt = TSOperator(solution, id_to_vehicle)
    TIMES = times
    for i in range(TIMES):
        epoch_start = time.time()
        opr.inter_exchange(w[0]*t)  # 路径间 交换
        opr.reinsert(w[1]*t)  # 路径间 移动订单

        opr.rearrange(w[2] * t)  # 路径内 重新安排订单送货点
        opr.intra_exchange(w[3]*t)  # 路径内 交换
        opr.intra_reinsert(w[2] * t)  # 路径内 重新插入订单

        epoch_used = time.time() - epoch_start
        time_used += epoch_used
        if time_used + epoch_used > 500:
            print('有超时风险，退出')
            break
    print('优化成功次数：{}  成功率{:.1f}%'.format(opr.success, 100*opr.success/sum(w)/TIMES/t))
    return opr.base_solution.vehicle_id_to_planned_route


def schedule_baseline3():
    # read the input json, you can design your own classes
    id_to_factory, id_to_unallocated_order_item, id_to_ongoing_order_item, id_to_vehicle = __read_input_json()

    # 构造初始解
    vehicle_id_to_destination = {}
    vehicle_id_to_planned_route = first_solution(
        id_to_unallocated_order_item,
        id_to_vehicle,
        id_to_factory)

    # create the output of the algorithm
    for vehicle_id, vehicle in id_to_vehicle.items():
        origin_planned_route = vehicle_id_to_planned_route.get(vehicle_id)
        # Combine adjacent-duplicated nodes.
        __combine_duplicated_nodes(origin_planned_route)

        destination = None
        planned_route = []
        # determine the destination
        if vehicle.destination is not None:
            if len(origin_planned_route) == 0:
                logger.error(f"Planned route of vehicle {vehicle_id} is wrong")
            else:
                destination = origin_planned_route[0]
                destination.arrive_time = vehicle.destination.arrive_time
                planned_route = [origin_planned_route[i] for i in range(1, len(origin_planned_route))]
        elif len(origin_planned_route) > 0:
            destination = origin_planned_route[0]
            planned_route = [origin_planned_route[i] for i in range(1, len(origin_planned_route))]

        vehicle_id_to_destination[vehicle_id] = destination
        vehicle_id_to_planned_route[vehicle_id] = planned_route

    # output the dispatch result
    __output_json(vehicle_id_to_destination, vehicle_id_to_planned_route)


def re_assign(id_to_unallocated_order_item: dict, id_to_vehicle: dict, id_to_factory: dict, id_to_all_myorder):
    vehicle_id_to_planned_route = {}  # 存放每辆车的路径规划（MyNode list）
    pre_matching_item_ids = []
    pre_matching_order_ids = []
    # 处理车辆身上已经装载的货物 ongoing
    for vehicle_id, vehicle in id_to_vehicle.items():
        unloading_sequence_of_items = vehicle.get_unloading_sequence()  # 卸货序列（也就是carry_items的倒序）
        vehicle_id_to_planned_route[vehicle_id] = []
        if len(unloading_sequence_of_items) > 0:
            if len(vehicle.destination.delivery_items) < 1:
                pickup_items = vehicle.destination.pickup_items
                order_id = pickup_items[0].order_id
                delivery_id = pickup_items[0].delivery_factory_id
                pickup_items = [item for item in pickup_items if item.delivery_factory_id == delivery_id]
                pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(pickup_items, id_to_factory)
                pickup_node.set_order_index(id_to_all_myorder[order_id].order_index)  # 设置Node的 order index，用于检查LIFO
                delivery_node.set_order_index(id_to_all_myorder[order_id].order_index)
                vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
                vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
                pre_matching_item_ids.extend([item.id for item in pickup_items])
                pre_matching_order_ids.append(pickup_items[0].order_id)

            delivery_item_list = []
            factory_id = unloading_sequence_of_items[0].delivery_factory_id  # 卸货序列第一个货物的目的工厂
            for item in unloading_sequence_of_items:
                if item.delivery_factory_id == factory_id:  # 清点货物，去同个工厂的放在一起
                    delivery_item_list.append(item)
                else:  # 遇到不同工厂的了，先把之前的那些打包成一个节点
                    factory = id_to_factory.get(factory_id)
                    node = MyNode(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                    node.set_order_index(id_to_all_myorder[item.order_id].order_index)
                    vehicle_id_to_planned_route[vehicle_id].append(node)  # 把节点加入路线规划
                    delivery_item_list = [item]  # 开始新的工厂货物清点
                    factory_id = item.delivery_factory_id  # 标记新工厂id
            if len(delivery_item_list) > 0:  # 处理最后一个工厂的货物
                factory = id_to_factory.get(factory_id)
                node = MyNode(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                vehicle_id_to_planned_route[vehicle_id].append(node)

    # 处理已接单的空车，目标已确定，必须前往目标点（最多是加减订单，但不能减完）也就是对未分配订单中实际已确定为destination的做个特殊处理

    for vehicle_id, vehicle in id_to_vehicle.items():
        if vehicle.carrying_items.is_empty() and vehicle.destination is not None:  # 空车，且有下一个目标（说明是去取货）
            pickup_items = vehicle.destination.pickup_items
            order_id = pickup_items[0].order_id
            delivery_id = pickup_items[0].delivery_factory_id
            pickup_items = [item for item in pickup_items if item.delivery_factory_id == delivery_id]
            pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(pickup_items, id_to_factory)
            pickup_node.set_order_index(id_to_all_myorder[order_id].order_index)  # 设置Node的 order index，用于检查LIFO
            delivery_node.set_order_index(id_to_all_myorder[order_id].order_index)
            vehicle_id_to_planned_route[vehicle_id].append(pickup_node)
            vehicle_id_to_planned_route[vehicle_id].append(delivery_node)
            pre_matching_item_ids.extend([item.id for item in pickup_items])
            pre_matching_order_ids.append(pickup_items[0].order_id)

    # 构造 {未分派订单id：货物列表} 字典
    order_id_to_items = {}
    for item_id, item in id_to_unallocated_order_item.items():  # 遍历每个货物
        if item_id in pre_matching_item_ids:  # 已经分配，车子在路上了， 不能再分配了 (如果是合单运送，也可以重分配但没必要)
            continue
        order_id = item.order_id
        if order_id not in order_id_to_items:
            order_id_to_items[order_id] = []
        order_id_to_items[order_id].append(item)

    order_id_to_items = merge_order(order_id_to_items)  # 合并订单

    return order_id_to_items, vehicle_id_to_planned_route

'''
*** 数据读取与处理 相关函数 ***
'''


def __read_input_json():
    """
    读取输入信息json文件
    :return: dict格式的工厂、订单、车辆信息
    """
    # read the factory info
    id_to_factory = get_factory_info(Configs.factory_info_file_path)

    # read the route map
    # code_to_route = get_route_map(Configs.route_info_file_path)
    # route_map = Map(code_to_route)

    # read the input json, you can design your own classes
    unallocated_order_items = read_json_from_file(Configs.algorithm_unallocated_order_items_input_path)
    id_to_unallocated_order_item = get_order_item_dict(unallocated_order_items, 'OrderItem')

    ongoing_order_items = read_json_from_file(Configs.algorithm_ongoing_order_items_input_path)
    id_to_ongoing_order_item = get_order_item_dict(ongoing_order_items, 'OrderItem')

    id_to_order_item = {**id_to_unallocated_order_item, **id_to_ongoing_order_item}

    vehicle_infos = read_json_from_file(Configs.algorithm_vehicle_input_info_path)
    id_to_vehicle = get_vehicle_instance_dict(vehicle_infos, id_to_order_item, id_to_factory)

    return id_to_factory, id_to_unallocated_order_item, id_to_ongoing_order_item, id_to_vehicle


def __output_json(vehicle_id_to_destination, vehicle_id_to_planned_route):
    write_json_to_file(Configs.algorithm_output_destination_path, convert_nodes_to_json(vehicle_id_to_destination))
    write_json_to_file(Configs.algorithm_output_planned_route_path, convert_nodes_to_json(vehicle_id_to_planned_route))


def __create_pickup_and_delivery_nodes_of_items(items: list, id_to_factory: dict):
    """
    将items打包成对应的取送货Node
    :param items: 待打包的货物列表
    :param id_to_factory: 工厂信息字典
    :return: 取货节点，送货节点
    """
    pickup_factory_id = __get_pickup_factory_id(items)
    delivery_factory_id = __get_delivery_factory_id(items)
    if len(pickup_factory_id) == 0 or len(delivery_factory_id) == 0:
        return None, None

    pickup_factory = id_to_factory.get(pickup_factory_id)
    delivery_factory = id_to_factory.get(delivery_factory_id)
    pickup_node = MyNode(pickup_factory.id, pickup_factory.lng, pickup_factory.lat, copy.copy(items), [])

    delivery_items = []
    last_index = len(items) - 1
    for i in range(len(items)):
        delivery_items.append(items[last_index - i])
    delivery_node = MyNode(delivery_factory.id, delivery_factory.lng, delivery_factory.lat, [], copy.copy(delivery_items))
    return pickup_node, delivery_node


def __get_pickup_factory_id(items):
    if len(items) == 0:
        logger.error("Length of items is 0")
        return ""

    factory_id = items[0].pickup_factory_id
    for item in items:
        if item.pickup_factory_id != factory_id:
            logger.error("The pickup factory of these items is not the same")
            return ""

    return factory_id


def __get_delivery_factory_id(items):
    if len(items) == 0:
        logger.error("Length of items is 0")
        return ""

    factory_id = items[0].delivery_factory_id
    for item in items:
        if item.delivery_factory_id != factory_id:
            logger.error("The delivery factory of these items is not the same")
            return ""

    return factory_id


# 合并相邻重复节点 Combine adjacent-duplicated nodes.
def __combine_duplicated_nodes(nodes):
    n = 0
    while n < len(nodes)-1:
        if nodes[n].id == nodes[n+1].id:
            nodes[n].delivery_items.extend(nodes[n+1].delivery_items)
            nodes[n].pickup_items.extend(nodes.pop(n+1).pickup_items)
        else:
            n += 1

def merge_order(order_id_to_items):
    new_dict = {}
    keys = list(order_id_to_items.keys())
    remove_keys = []
    while len(keys) > 0:
        order_id = keys[0]
        keys.remove(order_id)
        new_dict[order_id] = order_id_to_items[order_id]
        p = order_id_to_items[order_id][0].pickup_factory_id
        d = order_id_to_items[order_id][0].delivery_factory_id
        demand = calculate_demand(new_dict[order_id])
        for key in keys:
            key_p = order_id_to_items[key][0].pickup_factory_id
            key_d = order_id_to_items[key][0].delivery_factory_id
            if p == key_p and d == key_d:
                if calculate_demand(order_id_to_items[key]) + demand < MAX_CAPACITY:
                    new_dict[order_id].extend(order_id_to_items[key])
                    remove_keys.append(key)
                    demand = calculate_demand(new_dict[order_id])
                    # print('***** 合并 {}  into {} ***** '.format(key, order_id))
                    # print('***** 合并 {}  into {} ***** ({} -> {})'.format(key, order_id, calculate_demand(order_id_to_items[key]), demand))
        for k in remove_keys:
            keys.remove(k)
        remove_keys = []

    return new_dict