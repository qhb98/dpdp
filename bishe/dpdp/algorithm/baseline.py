import copy
import pickle
import os
import numpy as np

from .utils import get_node_matrix, is_begin

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
ALPHA = 200
save_path = os.path.join(Configs.algorithm_folder_path, 'order_dict.pkl')


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
        self.over_time = 0  # 该解的预估超时时间
        self.total_dis = 0  # 该解的预估总路程
        self.cost = 99999999  # 代价 = 总超时小时数*10000 + 车辆平均路程

    def get_over_time(self):
        """
        计算超时时间
        :return:
        """
        pass

    def init_time_window(self, id_to_vehicle):
        """
        计算各个路径点的[arrive, leave]时间窗
        :return:
        """
        for vid, route in self.vehicle_id_to_planned_route.items():
            self.time_window[vid] = []
            self.space[vid] = []
            FIRST = True
            for idx in range(len(route)):
                if FIRST:
                    # print('fisrt node')
                    # 首个节点两种情况，1）节点已经在当前destination了 2）车子当前为停车状态，节点是新规划的还没传入模拟器
                    # 已经为destination，直接取车辆信息中的到达时间
                    if id_to_vehicle[vid].destination is not None:
                        arrive_time = id_to_vehicle[vid].destination.arrive_time
                        leave_time = id_to_vehicle[vid].destination.leave_time
                    else:  # 新规划的，需要自己算时间
                        v = id_to_vehicle[vid]  # Vehicle object
                        now = node_str2int[v.cur_factory_id]
                        next = route[idx].int_id
                        arrive_time = v.leave_time_at_current_factory + node_matrix[now][next]
                        leave_time = arrive_time + DOCK_TIME + route[idx].service_time
                    # 计算初始剩余容量
                    space = get_vehicle_space(id_to_vehicle[vid])
                    FIRST = False
                else:
                    now = route[idx-1].int_id
                    next = route[idx].int_id
                    arrive_time = leave_time + node_matrix[now][next]
                    leave_time = arrive_time + DOCK_TIME + route[idx].service_time
                # 计算剩余容量
                unload_demand = calculate_demand(route[idx].delivery_items)
                load_demand = calculate_demand(route[idx].pickup_items)
                space = space + unload_demand - load_demand

                self.space[vid].append(space)
                self.time_window[vid].append([arrive_time, leave_time])

    def check(self):
        """
        检查是否满足LIFO约束及容量约束
        :return:
        """
        pass

    def get_cost(self):
        """
        计算cost
        :return:
        """

    @ staticmethod
    def get_node_dis(node1, node2):
        node1_id = node1.int_id
        node2_id = node2.int_id
        time_dis = node_matrix[node1_id][node2_id]

        return time_dis


class MyOrder(object):
    def __init__(self, order_id: str, order_index: int, items: list):
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

        self.split = 0  # 被拆成了几份
        self.assigned_vehicle = []  # 被分配给了哪辆车

        self.real_completion_time = -1  # 负值代表还没送达，暂时不知道
        self.late_time = -1

    # 根据订单中每个货物的状态确定订单状态
    def update_state(self):
        if self.delivery_state == 3:
            return
        INI_STATE = 100
        min_state = INI_STATE
        for item in self.items:
            if item.delivery_state < min_state:
                min_state = item.delivery_state
        if min_state < INI_STATE:
            self.delivery_state = min_state


class MyNode(Node):
    def __init__(self, factory_id: str, lng: float, lat: float, pickup_item_list: list, delivery_item_list: list):
        super().__init__(factory_id, lng, lat, pickup_item_list, delivery_item_list)
        self.order_index = 0  # N代表第N个订单取货节点，-N代表第N个订单送货节点
        self.is_fixed = False  # 被作为destination以后就算被固定了
        self.int_id = node_str2int[factory_id]

    def set_order_index(self, index):
        if len(self.pickup_items) > 0:
            self.order_index = index
        else:
            self.order_index = -1 * index


def first_solution(id_to_unallocated_order_item: dict, id_to_vehicle: dict, id_to_factory: dict):
    vehicle_id_to_destination = {}
    vehicle_id_to_planned_route = {}  # 存放每辆车的路径规划（MyNode list）

    vehicle_status = {}  # dict {车辆id: 状态} -> 0:停车等待；1：途中；2：节点工作（装卸货/靠台）
    # node_matrix, node_int2str, node_str2int = get_node_matrix()
    for vehicle_id in id_to_vehicle:
        MAX_CAPACITY = id_to_vehicle[vehicle_id].board_capacity
        break

    if is_begin(id_to_vehicle):
        id_to_all_myorder = {}
    elif os.path.exists(save_path):  # 要考虑下怎么处理不同case切换时的情况
        id_to_all_myorder = pickle.load(open(save_path, 'rb'))

    # pickle.dump(order_id_to_items, open('MyOrder_dict.pkl', 'wb'))  # 保存成pkl文件

    # 更新旧订单状态
    for order_id, myorder in id_to_all_myorder.items():
        myorder.update_state()

    # 处理车辆身上已经装载的货物
    for vehicle_id, vehicle in id_to_vehicle.items():
        unloading_sequence_of_items = vehicle.get_unloading_sequence()  # 卸货序列（也就是carry_items的倒序）
        vehicle_id_to_planned_route[vehicle_id] = []
        if len(unloading_sequence_of_items) > 0:
            delivery_item_list = []
            factory_id = unloading_sequence_of_items[0].delivery_factory_id  # 卸货序列第一个货物的目的工厂
            for item in unloading_sequence_of_items:
                if item.delivery_factory_id == factory_id:  # 清点货物，去同个工厂的放在一起
                    delivery_item_list.append(item)
                else:  # 遇到不同工厂的了，先把之前的那些打包成一个节点
                    factory = id_to_factory.get(factory_id)
                    node = MyNode(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                    vehicle_id_to_planned_route[vehicle_id].append(node)  # 把节点加入路线规划
                    delivery_item_list = [item]  # 开始新的工厂货物清点
                    factory_id = item.delivery_factory_id  # 标记新工厂id
            if len(delivery_item_list) > 0:  # 处理最后一个工厂的货物
                factory = id_to_factory.get(factory_id)
                node = MyNode(factory_id, factory.lng, factory.lat, [], copy.copy(delivery_item_list))
                vehicle_id_to_planned_route[vehicle_id].append(node)

    # 处理已接单的空车，目标已确定，必须前往目标点（最多是加减订单，但不能减完）也就是对未分配订单中实际已确定为destination的做个特殊处理
    pre_matching_item_ids = []
    pre_matching_order_ids = []
    for vehicle_id, vehicle in id_to_vehicle.items():
        if vehicle.carrying_items.is_empty() and vehicle.destination is not None:  # 空车，且有下一个目标（说明是去取货）
            pickup_items = vehicle.destination.pickup_items
            order_id = pickup_items[0].order_id
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

    for order_id, item_list in order_id_to_items.items():
        if order_id not in id_to_all_myorder:  # 新订单货物
            idx = len(id_to_all_myorder) + 1
            id_to_all_myorder[order_id] = MyOrder(order_id, idx, item_list)

    # 现在订单都在 {id: MyOrder object} 字典里了
    solution_list = []
    for order_id, items in order_id_to_items.items():  # 依次查看决策每个订单
        # 超过最大容量 需要拆分
        if id_to_all_myorder[order_id].demand > MAX_CAPACITY:
            # print('开始拆分', order_id, [x.id for x in items])
            pickup_node_id = items[0].pickup_factory_id  # 获取起点id
            delivery_node_id = items[0].delivery_factory_id
            item_idx = 0
            while item_idx < len(items):
                min_dis = 99999999
                min_vid = ''
                best_tmp_list = []
                temp_solution = Solution(copy.deepcopy(vehicle_id_to_planned_route))  # 临时解
                temp_solution.init_time_window(id_to_vehicle)
                for vehicle_id, vehicle in id_to_vehicle.items():
                    tmp_list = []
                    tmp_idx = item_idx
                    # 取该车规划路径最后一个点，比较时间距离（行驶+等待）
                    if len(vehicle_id_to_planned_route[vehicle_id]) > 0:
                        last_node_id = vehicle_id_to_planned_route[vehicle_id][-1].id
                        last_node_int_id = node_str2int[last_node_id]
                        last_node_space = temp_solution.space[vehicle_id][-1]
                        last_leave_time = temp_solution.time_window[vehicle_id][-1][1]  # 路径最后一个点的离开时间
                    else:  # 空车状态
                        last_node_id = id_to_vehicle[vehicle_id].cur_factory_id
                        last_node_int_id = node_str2int[last_node_id]
                        last_node_space = MAX_CAPACITY
                        last_leave_time = id_to_vehicle[vehicle_id].leave_time_at_current_factory
                        # 看看能塞进多少个，塞满为止
                    # print(vehicle_id, ' space: ', last_node_space)
                    while tmp_idx < len(items) and last_node_space - items[tmp_idx].demand > 0:
                        last_node_space -= items[tmp_idx].demand
                        tmp_list.append(items[tmp_idx])
                        tmp_idx += 1
                    # 构造节点
                    if len(tmp_list) < 1:
                        temp_solution.init_time_window(id_to_vehicle)
                    pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(tmp_list,
                                                                                             id_to_factory)  # 打包成Node
                    # 取当前订单取送货点
                    pickup_node_int_id = node_str2int[pickup_node_id]
                    delivery_node_int_id = node_str2int[delivery_node_id]
                    t_dis = node_matrix[last_node_int_id][pickup_node_int_id]  # 行驶时间距离 (最后一个点前往取货点)
                    t_delivery = pickup_node.service_time + node_matrix[pickup_node_int_id][delivery_node_int_id] \
                                 + DOCK_TIME + delivery_node.service_time  # 取送货整个完成时间
                    # 预估超时时间
                    # last_leave_time = temp_solution.time_window[vehicle_id][-1][1]  # 路径最后一个点的离开时间
                    over_time = (last_leave_time + t_dis + t_delivery) - tmp_list[0].committed_completion_time
                    over_time_cost = ALPHA * max(over_time, 0)
                    t_dis += over_time_cost  # 考虑超时惩罚

                    if t_dis < min_dis:
                        min_dis = t_dis
                        min_vid = vehicle_id
                        best_tmp_list = tmp_list

                # 更新最佳情况的计数器
                item_idx += len(best_tmp_list)
                # 选定插入车辆，更新路径
                id_to_all_myorder[order_id].split += 1
                split = id_to_all_myorder[order_id].split
                # 构造节点
                pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(best_tmp_list, id_to_factory)
                pickup_node.set_order_index(id_to_all_myorder[order_id].order_index+split*100000)  # order index，可检查LIFO
                delivery_node.set_order_index(id_to_all_myorder[order_id].order_index+split*100000)
                vehicle_id_to_planned_route[min_vid].append(pickup_node)
                vehicle_id_to_planned_route[min_vid].append(delivery_node)
                # print('插入', min_vid, calculate_demand(best_tmp_list), [x.id for x in best_tmp_list])

        else:  # 不用拆分，基于规则选一辆车插到最后面
            pickup_node, delivery_node = __create_pickup_and_delivery_nodes_of_items(items, id_to_factory)  # 打包成Node
            pickup_node.set_order_index(id_to_all_myorder[order_id].order_index)  # 设置Node的 order index，用于检查LIFO
            delivery_node.set_order_index(id_to_all_myorder[order_id].order_index)
            # 一种最简单的思路，新订单路径直接放在某辆车路径的最后面，肯定满足各种约束
            min_dis = 99999999
            min_vid = ''
            temp_solution = Solution(copy.deepcopy(vehicle_id_to_planned_route))  # 临时解
            temp_solution.init_time_window(id_to_vehicle)
            for vehicle_id, vehicle in id_to_vehicle.items():
                # 取该车规划路径最后一个点，比较时间距离（行驶+等待）
                if len(vehicle_id_to_planned_route[vehicle_id]) > 0:
                    last_node_id = vehicle_id_to_planned_route[vehicle_id][-1].id
                    last_node_int_id = node_str2int[last_node_id]
                    last_leave_time = temp_solution.time_window[vehicle_id][-1][1]
                else:  # 空路径
                    last_node_id = id_to_vehicle[vehicle_id].cur_factory_id
                    last_node_int_id = node_str2int[last_node_id]
                    last_leave_time = id_to_vehicle[vehicle_id].leave_time_at_current_factory
                # 取当前订单取送货点
                pickup_node_int_id = node_str2int[pickup_node.id]
                delivery_node_int_id = pickup_node.int_id  # 两种获取方法
                t_dis = node_matrix[last_node_int_id][pickup_node_int_id]  # 行驶时间距离 (最后一个点前往取货点)
                t_delivery = pickup_node.service_time +  node_matrix[pickup_node_int_id][delivery_node_int_id] \
                             + DOCK_TIME + delivery_node.service_time  # 取送货整个完成时间
                # 预估超时时间
                # last_leave_time = temp_solution.time_window[vehicle_id][-1][1]  # 路径最后一个点的离开时间
                over_time = (last_leave_time + t_dis + t_delivery) - items[0].committed_completion_time
                over_time_cost = ALPHA * max(over_time, 0)
                t_dis += over_time_cost  # 考虑超时惩罚

                if t_dis < min_dis:
                    min_dis = t_dis
                    min_vid = vehicle_id
            # tmp_route = copy.deepcopy(vehicle_id_to_planned_route)
            # 更新路径
            vehicle_id_to_planned_route[min_vid].append(pickup_node)
            vehicle_id_to_planned_route[min_vid].append(delivery_node)

    pickle.dump(id_to_all_myorder, open(save_path, 'wb'))  # 保存成pkl文件
    return vehicle_id_to_planned_route


def schedule_baseline():
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


# 合并相邻重复节点 Combine adjacent-duplicated nodes. 只考虑 取-取；送-取 情况，没考虑 送-送 情况（baseline不会出现）
def __combine_duplicated_nodes(nodes):
    n = 0
    while n < len(nodes)-1:
        if nodes[n].id == nodes[n+1].id:
            nodes[n].pickup_items.extend(nodes.pop(n+1).pickup_items)
        n += 1
