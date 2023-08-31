import math
import random
import matplotlib.pyplot as plt
import numpy as np

SHARD_NUMBER = 100
TX_NUM = 20000
RESHARD_TIME = 2

server_num = 200
node_num = 2000  # 委员会大小
proposer_num = 20  # proposers大小
account_num = 20000
# shardnum = 100  # 分片数量
txsize = 112  # bytes 根据blockene
signsize = 32  # bytes
hashsize = 32  # bytes
signthreshold = 850  # 根据blockene
witnesssize = 2000  # tx/witness
accountsize = 56  # bytes
rootsize = 40  # bytes hash 32 + int 8
validatesign = 0.0003  # s/sign
fixedlatency = 0.0005  # s 传输时延
bandwidth = 1024 * 1024  # bytes/s 带宽
servergossiplatency = math.log(server_num)  # gossip 时延
nodegossiplatency = math.log(node_num)  # gossip 时延

witness_num = int(TX_NUM / witnesssize)
GSread = TX_NUM * 3 / 90000  # 3s/90000tx
GSupdate = TX_NUM * 9 / 90000  # 9s/90000tx


class ProBlock:
    def __init__(self, txs, txs1, hs):
        self.txs = txs
        self.valid_txs = txs1
        self.h = hs


def concurrentRatio():
    #     # 连续两次选到相同账户的交易的概率
    #     # 生成 1-20 每个数各 81 个
    #     numbers_1_to_20 = [num for num in range(1, 20001) for _ in range(8100)]

    # # 生成 21-2000 每个数各 1 个
    #     numbers_21_to_2000 = [num for num in range(
    #         20001, 200001) for _ in range(100)]

    # # 合并两个列表
    #     all_numbers = numbers_1_to_20 + numbers_21_to_2000
    acc = np.arange(account_num)
    acc_list = acc.tolist()
    # print(len(all_numbers))
    elements1 = random.choices(acc_list, k=TX_NUM)
    elements2 = random.choices(acc_list, k=TX_NUM)
    # elements3 = random.choices(all_numbers, k=TX_NUM)
    # uelements = list(set(elements1).union(set(elements2)))
    repeat_element = [j for j in elements2 if j in elements1]
    return len(repeat_element) / 20000


def propogate_txs_delay():
    # 下载交易，进行区块见证
    servergossiplatency = math.log(server_num) + random.normalvariate(0, 1)
    downloadtx = fixedlatency + txsize * TX_NUM / bandwidth
    # 上传witnesslist并提出proposal
    uploadtx = fixedlatency + hashsize * TX_NUM / bandwidth
    +servergossiplatency
    Phase1 = downloadtx + uploadtx
    return Phase1


def consensus_delay(shardnum):
    txlistsize = hashsize * TX_NUM * shardnum / witnesssize
    servergossiplatency = math.log(server_num) + random.normalvariate(0, 1)
    nodegossiplatency = math.log(node_num) + random.normalvariate(0, 1)
    # 下载各个proposer提出的proposal
    downloadtxlist = fixedlatency + proposer_num * txlistsize / bandwidth
    # 假设有一半的proposer的witness达到了850个，下载并验证winner的区块见证
    downloadwitness = (
        fixedlatency
        + proposer_num / 2 * (signsize * signthreshold) / bandwidth
        + signthreshold * validatesign
    )
    # 处理跨分片交易：锁账户, 收集交易验证的结果
    downloadaccount = account_num * accountsize / bandwidth
    downloadroot = hashsize * shardnum / bandwidth
    # BBA共识，将符合要求的proposal上传，最后只要通知结果，即是否达成了共识，所以是8bytes
    BBA = (
        fixedlatency
        + servergossiplatency
        + (txlistsize + accountsize * account_num + hashsize * shardnum) / bandwidth
        + fixedlatency
        + nodegossiplatency
        + 8 / bandwidth
    )
    Phase2 = downloadtxlist + downloadwitness
    +downloadaccount + downloadroot + BBA
    return Phase2


def validate_delay(CroRatio):
    # 使用blockene中的GSread和GSupdate更新状态
    servergossiplatency = math.log(server_num) + random.normalvariate(0, 1)
    validate = 2 * GSread + validatesign * 2 * TX_NUM + GSupdate  # 并发控制
    UploadSuperTx = CroRatio * TX_NUM * hashsize / bandwidth
    +servergossiplatency
    Phase3 = validate + UploadSuperTx
    # print(servergossiplatency)
    return Phase3


tps = []

for shard_number in range(10, SHARD_NUMBER + 1, 10):
    horizonchain = []

    i = 0
    CroRatio = (shard_number - 1) / shard_number
    txpool = [0 for i in range(shard_number)]
    txs_number = 0

    phase1 = 0
    phase2 = 0

    while len(horizonchain) < 30:
        i += 1
        t = i / 10
        # print(i, t)
        # 产生第一个区块时，系统中只有一个委员会在运行
        if len(horizonchain) == 0:
            if phase1 == 0:
                tx = 0
                phase1 = (
                    RESHARD_TIME + propogate_txs_delay() + consensus_delay(shard_number)
                )
                # print(phase1)
                for s in range(shard_number):
                    b = random.normalvariate(0, 0.1)
                    num = TX_NUM * (1 + b)
                    a = random.uniform(0, 1)
                    if a < 0.05:
                        num = 0
                    txpool[s] += num
                # print(TX_NUM)

            else:
                if phase1 < t:
                    for j in txpool:
                        tx += j
                    block = ProBlock(tx, 0, len(horizonchain) + 1)
                    horizonchain.append(block)
                    # print("添加第一個區塊")
                    # print(len(horizonchain))
                    txpool = [0 for i in range(shard_number)]
                    phase1 = 0
                    phase2 = 0
                    tx = 0
        else:
            if phase1 == 0:
                phase1 = (
                    t
                    + RESHARD_TIME
                    + propogate_txs_delay()
                    + consensus_delay(shard_number)
                )

                phase2 = t + validate_delay(CroRatio)
                print(phase1, phase2)
                for s in range(shard_number):
                    b = random.normalvariate(0, 0.1)
                    # 网络波动
                    num = TX_NUM * (1 + b)
                    #
                    a = random.uniform(0, 1)
                    if a < 0.05:
                        num = 0
                    txpool[s] += num
                    # print(TX_NUM)

            else:
                if phase2 != 0:
                    valid_txs = 0
                if phase2 < t:
                    valid_txs = horizonchain[len(horizonchain) - 1].txs * (
                        1 - concurrentRatio() - concurrentRatio() - concurrentRatio()
                    )
                    # print("交易验证成功", valid_txs)
                    phase2 = 0
                if phase1 < t and phase2 < t:
                    for j in txpool:
                        tx += j
                    block = ProBlock(tx, valid_txs, len(horizonchain) + 1)
                    horizonchain.append(block)
                    # print("交易见证，共识成功", tx)
                    txpool = [0 for i in range(shard_number)]
                    valid_txs = 0
                    phase1 = 0
                    tx = 0

    for block in horizonchain:
        txs_number += block.valid_txs

    t = i / 10
    print(shard_number, txs_number / t)
    tps.append(txs_number / t)

print(tps)
plt.xlabel("SHARD_NUMBER")
plt.ylabel("tps")

plt.plot(range(10, SHARD_NUMBER + 1, 10), tps, label="aaa", marker=".")

# plt.legend(loc="upper right")
plt.show()
