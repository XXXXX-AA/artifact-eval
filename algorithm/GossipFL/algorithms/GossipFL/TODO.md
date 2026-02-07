### 首先要在每次cluster更新时，通知所有cluster内的节点
要点是要保证所有的cluster内的节点是同步的。因此增加一个新的消息类型，和handler
#### （1）cluster合并
#### （2）cluster加入
#### （3）cluster离开

###   2025/6/6
#### 增加cluster广播



### 2025/6/24
#### 
#### 功能，coalition公式，-loss+bw+sim-RoundDist




## 2025/6/25
#### （1）增加本地buffer+senderID，用来存储邻居发来的参数→本来就有的，是
'''
self.worker.add_result(sender_id, msg_params)
    
def add_result(self, worker_index, updated_information):
    self.worker_result_dict[worker_index] = updated_information
    self.flag_neighbor_result_received_dict[worker_index] = True
'''


#### 增加的代码如下：
'''
BASEDECENT/decentralized_worker.py
self.flag_neighbor_result_received_dict_for_flock = deque()


    # for flock
    def add_result_for_flock(self, worker_index, updated_information):
        self.worker_result_dict[worker_index] = updated_information
        self.flag_neighbor_result_received_dict_for_flock.append(worker_index)



    # for flock
    def aggregate_for_flock(self):
        start_time = time.time()
        model_list = []
        training_weights = 0

        #  TODO, There are some bugs
        model_list.append((0.5, self.get_model_params()))

        if self.flag_neighbor_result_received_dict_for_flock:
            neighbor_idx = self.flag_neighbor_result_received_dict_for_flock.popleft()
            model_list.append((0.5, self.worker_result_dict[neighbor_idx]))
            training_weights += 0.5

        training_weights = 1.0
        logging.debug("len of self.worker_result_dict[idx] = " + str(len(self.worker_result_dict)))

        # logging.debug("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            # averaged_params[k] = averaged_params[k] * self.local_sample_number / training_num
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_weights
                if i == 0:
                    averaged_params[k] = local_model_params[k] * local_sample_number
                else:
                    averaged_params[k] += local_model_params[k] * local_sample_number
            averaged_params[k] /= training_weights

        end_time = time.time()
        logging.debug("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

'''

#### 改变SAPS的逻辑
##### 1、增加self.ifStop = False，在全部训练结束后通知coo
##### # 训练循环结束后 self.send_message_to_coordinator(MSG_TYPE_FINISHED)
##### 2、删除所有的

    self.start_epoch_event.clear()
    self.sync_receive_all_event.clear()
##### 3、每次训练完毕后不需要通知Coordinator，修改一下代码为：
    #self.test_and_send_to_coordinator(iteration, epoch)
    self.worker.test(self.epoch, self.test_tracker, self.metrics)

##### 4、在本worker结束所有epoch的训练之后，向coordinator发送结束信息，并继续监听，等待coordinator的结束信息。因此需要增加一个结束标志位
（1）增加结束标志位

    # whether all workers finised
    self.ifStop = False
（2）重写以下函数，当给coordinator发notification时，即代表本地训练结束，并等待全局结束信息
    def send_notify_to_coordinator(self, receive_id=0):
        logging.debug("send_notify_to_coordinator. receive_id = %s, round: %s" % (str(receive_id), str(self.global_round_idx)))
        message = Message(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR, self.get_sender_id(), receive_id)
        self.send_message(message)



### 2025-6-27
##### 改变了顺序，改变了aggregation、uncompress、uncompress_direct

### 2025-6-28 完成了异步训练。今天的任务是，local——gossip。应该怎么做呢？
1、每个节点应该有一个local视角。那么就是根据Gossipfl的矩阵，给每个节点单独的一行就行了。进行固定的存储。
2、coordinator需要生成这个矩阵，然后发送每一行给各个节点。
2、然后每个节点每轮随机选择。

