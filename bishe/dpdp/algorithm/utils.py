import copy
import os
import pandas as pd
import numpy as np
import random

from src.conf.configs import Configs
from src.common.stack import Stack

NO_DEALY_T = [6, 7, 8, 9, 10, 11, 12, 13, 30, 42, 43, 44, 55, 78, 109, 110, 111, 112, 113, 114]

class SearchOperator(object):
    def __init__(self, solution, id_to_vehicle):
        self.base_solution = solution
        self.new_solutions = []
        self.id_to_vehicle = id_to_vehicle
        self.k = len(solution.vehicle_id_to_planned_route)  # 几条路径
        self.success = 0


class ExchangeOperator(SearchOperator):
    """
    交换2对订单取送货点
    """
    def __init__(self, solution, id_to_vehicle):
        super().__init__(solution, id_to_vehicle)
        self.vid_list = [x for x in solution.vehicle_id_to_planned_route]

    def intra_exchange(self, n=None):
        """
        选取n条路径，进行内部的2个订单交换位置
        """
        if n is None:
            n = random.randint(1, self.base_solution.k)
        tmp_solution = copy.deepcopy(self.base_solution)
        for i in range(n):
            dis_mean = np.mean(list(self.base_solution.over_time.values()))
            tmp_vid_list = [x for x in self.vid_list if self.base_solution.over_time[x] > dis_mean]
            if len(tmp_vid_list) == 0:
                tmp_vid_list = self.vid_list
            vid = random.choice(tmp_vid_list)
            route = tmp_solution.vehicle_id_to_planned_route[vid]
            len_route = len(route)
            p_list = tmp_solution.pickup_index[vid]
            if len(p_list) < 2:
                continue
            up_bound = min(4, len(p_list))
            a_p = random.choice(p_list[:up_bound])
            b_p = random.choice(p_list)
            while a_p == b_p:
                b_p = random.choice(p_list)
            # (a_p, b_p) = random.sample(p_list, 2)

            # 寻找对应的送货点
            a_d = b_d = -1
            for idx in range(a_p+1, len_route):
                if route[idx].order_index + route[a_p].order_index == 0:
                    a_d = idx
                    break
            for idx in range(b_p+1, len_route):
                if route[idx].order_index + route[b_p].order_index == 0:
                    b_d = idx
                    break
            route[a_p], route[b_p] = route[b_p], route[a_p]
            route[a_d], route[b_d] = route[b_d], route[a_d]
            tmp_solution.vehicle_id_to_planned_route[vid] = route
            if not check_capacity_constraint(route, copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15):
                continue
            # 如果新解更好，替换旧解 [目前仅用超时情况衡量]
            # tmp_solution.calc_over_time(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
            tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
            if tmp_solution.total_cost+10 < self.base_solution.total_cost:
                # self.base_solution = copy.deepcopy(tmp_solution)
                # self.base_solution = tmp_solution
                self.base_solution.update_solution(tmp_solution, vid_list=[vid])
                # print('路径内部交换 发现更优解', vid)
                self.success += 1
            else:
                # tmp_solution = copy.deepcopy(self.base_solution)
                tmp_solution.update_solution(self.base_solution, vid_list=[vid])

    def inter_exchange(self, n=1):
        """
        选取n对路径，进行路径间订单交换
        """
        tmp_solution = copy.deepcopy(self.base_solution)
        for i in range(n):
            tmp_vid_list = [x for x in self.vid_list if len(tmp_solution.vehicle_id_to_planned_route[x]) > 2]
            if len(tmp_vid_list) < 2:
                continue
            (vid_a, vid_b) = random.sample(tmp_vid_list, 2)  # 挑选2条不同的路径
            route_a = tmp_solution.vehicle_id_to_planned_route[vid_a]
            route_b = tmp_solution.vehicle_id_to_planned_route[vid_b]
            # tmp_solution.calc_cost(vid_list=[vid_a, vid_b], update=True, id_to_vehicle=self.id_to_vehicle)  # 试试

            if len(route_a) - tmp_solution.mask[vid_a] < 2 or len(route_b) - tmp_solution.mask[vid_b] < 2:
                continue
            if len(tmp_solution.pickup_index[vid_a]) < 1 or len(tmp_solution.pickup_index[vid_b]) < 1:
                continue

            p_list = tmp_solution.pickup_index[vid_a]
            up_bound = min(4, len(p_list))
            idx_a_p = random.choice(p_list[0:up_bound])  # 订单a从靠前的位置选
            idx_b_p = random.choice(tmp_solution.pickup_index[vid_b])

            # 寻找对应的送货点
            idx_a_d = idx_b_d = -1
            for idx in range(idx_a_p + 1, len(route_a)):
                if route_a[idx].order_index + route_a[idx_a_p].order_index == 0:
                    idx_a_d = idx
                    break
            for idx in range(idx_b_p+1, len(route_b)):
                if route_b[idx].order_index + route_b[idx_b_p].order_index == 0:
                    idx_b_d = idx
                    break
            route_a[idx_a_p], route_b[idx_b_p] = route_b[idx_b_p], route_a[idx_a_p]
            route_a[idx_a_d], route_b[idx_b_d] = route_b[idx_b_d], route_a[idx_a_d]
            tmp_solution.vehicle_id_to_planned_route[vid_a] = route_a
            tmp_solution.vehicle_id_to_planned_route[vid_b] = route_b

            check = check_capacity_constraint(route_a, copy.deepcopy(self.id_to_vehicle[vid_a].carrying_items), 15) and \
                check_capacity_constraint(route_b, copy.deepcopy(self.id_to_vehicle[vid_b].carrying_items), 15)
            if not check:
                tmp_solution.update_solution(self.base_solution, vid_list=[vid_a, vid_b])
                continue

            tmp_solution.calc_cost(vid_list=[vid_a, vid_b], update=True, id_to_vehicle=self.id_to_vehicle)
            if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                self.base_solution.update_solution(tmp_solution, vid_list=[vid_a, vid_b])
                # print('路径间交换 发现更优解', vid_a, vid_b)
                self.success += 1
            elif i < n - 1:
                tmp_solution.update_solution(self.base_solution, vid_list=[vid_a, vid_b])

    def rearrange(self, n=1):
        """
        随机挑选n个取货点，重新安排它的送货点在路径中的位置
        :param n:
        :return:
        """
        # tabu = TabuTable(5)
        tmp_solution = copy.deepcopy(self.base_solution)
        for i in range(n):
            dis_mean = np.mean(list(self.base_solution.over_time.values()))
            tmp_vid_list = [x for x in self.vid_list if self.base_solution.over_time[x] > dis_mean]
            if len(tmp_vid_list) == 0:
                tmp_vid_list = self.vid_list
            vid = random.choice(tmp_vid_list)
            if len(self.base_solution.pickup_index[vid]) < 2:
                continue
            p_list = tmp_solution.pickup_index[vid][:-1]
            up_bound = min(4, len(p_list))
            p_list = p_list[:up_bound]
            if tmp_solution.vehicle_id_to_planned_route[vid][0].order_index > 0:
                p_list.append(0)
            idx_p = random.choice(p_list)  # 选一个取货点
            # route = copy.deepcopy(tmp_solution.vehicle_id_to_planned_route[vid])
            route = tmp_solution.vehicle_id_to_planned_route[vid]
            len_route = len(route)
            idx_d = -1
            for idx in range(idx_p + 1, len(route)):
                if route[idx].order_index + route[idx_p].order_index == 0:
                    idx_d = idx
                    break
            # 弹出
            node = route.pop(idx_d)
            s = Stack()
            best_idx = idx_d
            min_score = tmp_solution.total_cost
            bound = min(len_route, idx_p+9)
            for tmp_d in range(idx_p+1, bound):
                if s.is_empty() and tmp_d != idx_d: # LIFO约束上可以插入
                    route.insert(tmp_d, node)
                    # 检查空间约束
                    # check1 = check_space(route[idx_p+1:tmp_d+1], self.base_solution.space[vid][idx_p])
                    # check2 = check_capacity_constraint(route[:tmp_d+1], copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15)
                    # if check1 != check2:
                    #     raise ValueError('wrong !')
                    #     self.base_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
                    #     print(self.base_solution.space[vid])

                    if check_space(route[idx_p+1:tmp_d+1], self.base_solution.space[vid][idx_p]): # tmp_d+1
                    # if check_capacity_constraint(route[:tmp_d+1], copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15):
                        tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
                        if tmp_solution.total_cost+10 < min_score:
                            min_score = tmp_solution.total_cost
                            best_idx = tmp_d
                    else:
                        node = route.pop(tmp_d)
                        break
                    node = route.pop(tmp_d)
                if tmp_d == len_route-1:
                    break
                if not s.is_empty() and s.peek() + route[tmp_d].order_index == 0:
                    s.pop()
                else:
                    s.push(route[tmp_d].order_index)
            if best_idx != idx_d:
                route.insert(best_idx, node)
                tmp_solution.vehicle_id_to_planned_route[vid] = route
                tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
                if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                    self.base_solution.update_solution(tmp_solution, vid_list=[vid])
                    # print('重安排送货点 发现更优解', vid)
                    self.success += 1
                else:
                    tmp_solution.update_solution(self.base_solution, vid_list=[vid])
            else:
                tmp_solution.update_solution(self.base_solution, vid_list=[vid])

    def reinsert(self, n=1, intra=True):
        """
        将某条路径的订单插入到另一条路径
        """
        for i in range(n):
            tmp_solution = copy.deepcopy(self.base_solution)
            (vid_a, vid_b) = random.sample(self.vid_list, 2)  # 挑选2条不同的路径
            route_a = tmp_solution.vehicle_id_to_planned_route[vid_a]
            route_b = tmp_solution.vehicle_id_to_planned_route[vid_b]

            if len(route_a) < len(route_b) and random.random() < 0.8:  # 希望route_a是长的那条
                vid_a, vid_b = vid_b, vid_a
                route_a, route_b = route_b, route_a
            p_list = tmp_solution.pickup_index[vid_a]
            if len(p_list) < 1:
                continue
            improve = 0
            best_p = p_list[0]
            if random.random() < 0.6:
                for p_idx in p_list:
                    tmp_node_p = route_a.pop(p_idx)
                    d_idx = -1
                    for idx in range(p_idx, len(route_a)):
                        if route_a[idx].order_index + tmp_node_p.order_index == 0:
                            d_idx = idx
                            break
                    tmp_node_d = route_a.pop(d_idx)
                    tmp_solution.vehicle_id_to_planned_route[vid_a] = route_a
                    tmp_solution.calc_cost(vid_list=[vid_a], update=True, id_to_vehicle=self.id_to_vehicle)
                    if self.base_solution.total_cost - tmp_solution.total_cost > improve:
                        improve = self.base_solution.total_cost - tmp_solution.total_cost
                        best_p = p_idx
                    tmp_solution.update_solution(self.base_solution, vid_list=[vid_a])
                    route_a = tmp_solution.vehicle_id_to_planned_route[vid_a]
                # print(f'find best pickup node, improve {improve}')
            else:
                best_p = random.choice(p_list)

            idx_a_p = best_p
            node_a_p = route_a.pop(idx_a_p)  # 路线a的订单起点
            idx_a_d = -1
            for idx in range(idx_a_p, len(route_a)):
                if route_a[idx].order_index + node_a_p.order_index == 0:
                    idx_a_d = idx
                    break
            node_a_d = route_a.pop(idx_a_d)  # 路线a的订单终点

            idx_b_p = -1
            # 插入路线b
            len_route_b = len(route_b)
            # 看看有没有相同的取送货点可以合并
            flag = False
            for p_idx in tmp_solution.pickup_index[vid_b]:  # tmp_solution.pickup_index[vid_b]
                if route_b[p_idx].int_id == node_a_p.int_id:
                    idx_b_p = p_idx
                    if route_b[p_idx].service_time + node_a_p.service_time > 14*240:
                        continue
                    # 找到对应的送货点
                    d_idx = -1
                    for j in range(p_idx + 1, len(route_b)):
                        if route_b[j].order_index + route_b[p_idx].order_index == 0:
                            d_idx = j
                            break
                    if d_idx > 0 and route_b[d_idx].int_id == node_a_d.int_id:  # 找到了，尝试合并
                        tmp_route_b = copy.deepcopy(route_b)
                        tmp_route_b[p_idx].pickup_items.extend(node_a_p.pickup_items)
                        tmp_route_b[d_idx].delivery_items = node_a_d.delivery_items + tmp_route_b[d_idx].delivery_items
                        tmp_route_b[p_idx].update_service_time()
                        tmp_route_b[d_idx].update_service_time()
                        tmp_route_b[p_idx].merge = True
                        tmp_route_b[d_idx].merge = True
                        if not check_capacity_constraint(tmp_route_b, copy.deepcopy(self.id_to_vehicle[vid_b].carrying_items), 15):
                            continue
                        tmp_solution.vehicle_id_to_planned_route[vid_b] = tmp_route_b
                        tmp_solution.calc_cost(vid_list=[vid_a, vid_b], update=True, id_to_vehicle=self.id_to_vehicle)
                        if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                            self.base_solution.update_solution(tmp_solution, vid_list=[vid_a, vid_b])
                            print('路径间移动 成功合并 {}({}-{}) -> {} 发现更优解'.format(vid_a, idx_a_p, idx_a_d, vid_b))
                            self.success += 1
                            flag = True
                        tmp_solution.vehicle_id_to_planned_route[vid_b] = route_b
                        tmp_solution.calc_cost(vid_list=[vid_a, vid_b], update=True, id_to_vehicle=self.id_to_vehicle)

            if flag:  # 合并成功，肯定比随机插入要好，后续就不尝试了
                continue

            for p_idx in range(0, len(route_b)):
                if route_b[p_idx].int_id == node_a_p.int_id:
                    idx_b_p = p_idx
                    break
            # 随便找个位置插入
            if idx_b_p == -1:
                idx_b_p = random.choice(range(tmp_solution.mask[vid_b], len_route_b+1))
            else:
                idx_b_p += 1
            route_b.insert(idx_b_p, node_a_p)
            route_b.insert(idx_b_p+1, node_a_d)
            tmp_solution.vehicle_id_to_planned_route[vid_a] = route_a
            tmp_solution.vehicle_id_to_planned_route[vid_b] = route_b

            if len_route_b == 0:
                cur_space = 15
            else:
                cur_space = self.base_solution.space[vid_b][idx_b_p-1]

            if not check_space(route_b[idx_b_p: idx_b_p+1], cur_space):
                continue
            tmp_solution.calc_cost(vid_list=[vid_a, vid_b], update=True, id_to_vehicle=self.id_to_vehicle)
            if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                self.base_solution.update_solution(tmp_solution, vid_list=[vid_a, vid_b])
                # print('路径间移动订单 {}({}-{}) -> {} 发现更优解'.format(vid_a, idx_a_p, idx_a_d, vid_b))
                self.success += 1

    def intra_reinsert(self, n=5):
        """
        挑选一条路径中的一个订单，重新插入/合并该订单的位置
        :param n:
        :return:
        """
        tmp_solution = copy.deepcopy(self.base_solution)
        for i in range(n):
            dis_mean = np.mean(list(self.base_solution.over_time.values()))
            tmp_vid_list = [x for x in self.vid_list if self.base_solution.over_time[x] > dis_mean]
            if len(tmp_vid_list) == 0:
                tmp_vid_list = self.vid_list

            vid = random.choice(tmp_vid_list)
            route = tmp_solution.vehicle_id_to_planned_route[vid]
            p_list = tmp_solution.pickup_index[vid]
            if len(p_list) < 2:
                continue
            mid = min(len(p_list), 4) * -1
            idx_p = random.choice(p_list[mid:])
            node_p = route.pop(idx_p)  # 路线a的订单起点
            idx_d = -1
            for idx in range(idx_p, len(route)):
                if route[idx].order_index + node_p.order_index == 0:
                    idx_d = idx
                    break
            node_d = route.pop(idx_d)  # 路线a的订单终点

            # 看看有没有相同的取送货点可以合并
            flag = False
            for p_idx in range(tmp_solution.mask[vid], len(route)):  # tmp_solution.pickup_index[vid_b]
                if route[p_idx].order_index > 0 and route[p_idx].int_id == node_p.int_id:
                    if route[p_idx].service_time + node_p.service_time > 15*240:
                        continue
                    # 找到对应的送货点
                    d_idx = -1
                    for j in range(p_idx + 1, len(route)):
                        if route[j].order_index + route[p_idx].order_index == 0:
                            d_idx = j
                            break
                    if d_idx > 0 and route[d_idx].int_id == node_d.int_id:  # 找到了，尝试合并
                        tmp_route = copy.deepcopy(route)
                        tmp_route[p_idx].pickup_items.extend(node_p.pickup_items)
                        tmp_route[d_idx].delivery_items = node_d.delivery_items + tmp_route[d_idx].delivery_items
                        tmp_route[p_idx].update_service_time()
                        tmp_route[d_idx].update_service_time()
                        tmp_route[p_idx].merge = True
                        tmp_route[d_idx].merge = True
                        if not check_capacity_constraint(tmp_route, copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15):
                            continue
                        tmp_solution.vehicle_id_to_planned_route[vid] = tmp_route
                        tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
                        if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                            self.base_solution.update_solution(tmp_solution, vid_list=[vid])
                            print('>>> 路径内 成功合并 {} '.format(vid))
                            self.success += 1
                            flag = True
                        tmp_solution.vehicle_id_to_planned_route[vid] = route
                        tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)

            if flag:  # 合并成功，不尝试了
                tmp_solution.update_solution(self.base_solution, vid_list=[vid])
                continue

            # 看看有没有相同节点的，可以安排挨着
            idx_p = -1
            for p_idx in range(len(route)-1, 0, -1):
                if route[p_idx].int_id == node_p.int_id:
                    idx_p = p_idx
                    break

            if idx_p == -1:
                idx_p = random.choice(range(tmp_solution.mask[vid], len(route)+1))
            else:
                idx_p += 1
            route.insert(idx_p, node_p)
            route.insert(idx_p+1, node_d)
            tmp_solution.vehicle_id_to_planned_route[vid] = route

            if not check_capacity_constraint(route, copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15):
                tmp_solution.update_solution(self.base_solution, vid_list=[vid])
                continue
            tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
            if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                self.base_solution.update_solution(tmp_solution, vid_list=[vid])
                # print('路径内重新插入{} 发现更优解'.format(vid))
                self.success += 1

            tmp_solution.update_solution(self.base_solution, vid_list=[vid])

class TSOperator(SearchOperator):
    """
    opt相关的搜索算子
    """
    def __init__(self, solution, id_to_vehicle):
        super().__init__(solution, id_to_vehicle)
        self.vid_list = [x for x in solution.vehicle_id_to_planned_route]

    def two_opt(self, n):
        for i in range(n):
            vid = random.choice(self.vid_list)
            tmp_solution = copy.deepcopy(self.base_solution)
            route = tmp_solution.vehicle_id_to_planned_route[vid]
            if len(route) < 4:
                continue
            start = self.base_solution.mask[vid]
            p_list = [x for x in tmp_solution.pickup_index[vid][start:] if x < 8]
            if len(p_list) < 1:
                continue
            end = random.choice(p_list)
            while route[start].order_index < 0:
                start += 1
            route[start: end] = route[end-1: start-1: -1]
            check = check_LIFO(route) and check_capacity_constraint(route, copy.deepcopy(self.id_to_vehicle[vid].carrying_items), 15)
            if not check:
                continue
            tmp_solution.vehicle_id_to_planned_route[vid] = route
            tmp_solution.calc_cost(vid_list=[vid], update=True, id_to_vehicle=self.id_to_vehicle)
            if tmp_solution.total_cost + 10 < self.base_solution.total_cost:
                self.base_solution.update_solution(tmp_solution, vid_list=[vid])
                print('2-opt 发现更优解', vid)
                self.success += 1


class TabuTable(object):
    def __init__(self, size=5):
        self.table = []
        self.max_size = size

    def check(self, x):
        if x in self.table:
            return False
        return True

    def add(self, x):
        if len(self.table) >= self.max_size:
            self.table.pop(0)
        self.table.append(x)


# 自定义list/dict的copy函数, 比copy.deepcopy快
_dispatcher = {}
def _copy_list(_l):
    ret = _l.copy()
    for idx, item in enumerate(ret):
        cp = _dispatcher.get(type(item))
        if cp is not None:
            ret[idx] = cp(item)
    return ret
_dispatcher[list] = _copy_list


def _copy_dict(d):
    ret = d.copy()
    for key, value in ret.items():
        cp = _dispatcher.get(type(value))
        if cp is not None:
            ret[key] = cp(value)

    return ret
_dispatcher[dict] = _copy_dict


def mydeepcopy(sth):
    cp = _dispatcher.get(type(sth))
    if cp is None:
        return sth
    else:
        return cp(sth)


def get_node_matrix():
    """
    获取工厂节点矩阵（时间）
    :return:
    """
    node_str2int = {}  # str id -> int id
    node_int2str = []  # int id -> str id

    factory_info_file_path = os.path.join(Configs.benchmark_folder_path, Configs.factory_info_file)
    route_info_file_path = os.path.join(Configs.benchmark_folder_path, Configs.route_info_file)
    df_factory = pd.read_csv(factory_info_file_path)
    df_route = pd.read_csv(route_info_file_path)
    for index, row in df_factory.iterrows():
        factory_id_str = str(row['factory_id'])
        node_int2str.append(factory_id_str)
        node_str2int[factory_id_str] = index
    node_num = len(node_int2str)
    node_matrix = np.zeros((node_num, node_num))
    for index, row in df_route.iterrows():
        # route_code = str(row['route_code'])
        start_factory_id = str(row['start_factory_id'])
        end_factory_id = str(row['end_factory_id'])
        # distance = float(row['distance'])
        transport_time = int(row['time'])
        start_id_int = node_str2int[start_factory_id]
        end_id_int = node_str2int[end_factory_id]
        node_matrix[start_id_int][end_id_int] = transport_time
    print('init graph done')

    return node_matrix, node_int2str, node_str2int


def is_begin(id_to_vehicle):
    for v_id, vehicle in id_to_vehicle.items():
        if vehicle.gps_update_time != vehicle.arrive_time_at_current_factory+600:
            return False
        if vehicle.destination is not None:
            return False
        if not vehicle.carrying_items.is_empty():
            return False

    return True


def check_capacity_constraint(route: list, carrying_items, max_capacity):
    left_capacity = max_capacity

    # 根据当前携带的物品，计算当前剩余空间
    while not carrying_items.is_empty():
        item = carrying_items.pop()
        left_capacity -= item.demand
        if left_capacity < 0:
            return False

    for node in route:
        delivery_items = node.delivery_items
        pickup_items = node.pickup_items
        for item in delivery_items:
            left_capacity += item.demand
            if left_capacity > max_capacity:
                return False

        for item in pickup_items:
            left_capacity -= item.demand
            if left_capacity < 0:
                return False

    return True


def check_space(route, current_space, max_capacity=15):
    left_capacity = current_space

    for node in route:
        delivery_items = node.delivery_items
        pickup_items = node.pickup_items
        for item in delivery_items:
            left_capacity += item.demand
            if left_capacity > max_capacity:
                return False

        for item in pickup_items:
            left_capacity -= item.demand
            if left_capacity < 0:
                return False

    return True


def check_LIFO(route: list):
    s = Stack()
    for node in route:
        if s.is_empty():
            s.push(node.order_index)
            continue
        if node.order_index + s.peek() == 0:
            s.pop()

    return s.is_empty()