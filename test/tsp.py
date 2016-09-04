#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def generateDistMatrix(distlist):
    #总共有31*15=465个值
    distmtx = np.zeros((31,31))
    count = 0
    for i in range(31):
        for j in range(i+1,31):
            count += 1
            distmtx[i,j]=distmtx[j,i]=distlist[count-1]
    return distmtx

def tsp_dist(dist, path):  #求路径path对应的距离
    #path是长度为n的向量,包含1-N的整数
    n= len(path)
    length = 0
    for i in range(n-1):
        length = length + dist[path[i]-1, path[i+1]-1]
    length = length + dist[path[0]-1, path[n-1]-1]
    return length

def tsp_new_path(old_path):
    #在oldpath附近产生新的路径
    n = len(old_path)
    new_path = old_path[:]
    if np.random.rand()>0.25:  #随机生成2个位置,并交换
        positions = np.ceil(np.random.rand(1,2)*n).tolist()[0]
        a = int(positions[0])-1
        b = int(positions[1])-1
        if a != b:
            new_path[a] = old_path[b]
            new_path[b] = old_path[a]
        else:
            new_path = tsp_new_path(old_path)
    else:  #否则生成三个位置,交换a-b和b-c段
        positions = np.ceil(np.random.rand(1,3)*n).tolist()[0]
        ordered_position = positions[:]
        ordered_position.sort()
        a = int(ordered_position[0])
        b = int(ordered_position[1])
        c = int(ordered_position[2])
        if a!= b and b!= c:
            new_path[a:c] = old_path[b:c]+old_path[a:b]
        else:
            new_path = tsp_new_path(old_path)
    return new_path

distlist = [1078.2344, 118.8984, 263.3544, 398.18, 401.5136, 634.1248, 866.3656, 1060.8256, 366.696, 904.5168,
            901.924, 1134.9056, 1254.9152, 1567.533, 1729.398, 626.3464, 1051.936, 1343.07, 1893.4848, 2286.4792,
            2049.7936, 909.332, 878.9592, 1178.2424, 1321.2168, 2399.0808, 1511.232, 1731.62, 2085.352, 2557.9824,
            962.6696, 988.968, 1095.6432, 1381.592, 1188.984, 1450.8568, 1679.0232, 735.244, 268.9104, 398.5504,
            160.3832, 600.7888, 605.9744, 686.3512, 824.5104, 684.8696, 880.4408, 1210.4672, 1665.6888, 1599.3872,
            1217.5048, 1597.1648, 1717.1744, 1908.6712, 3265.076, 1655.3176, 1522.344, 1956.0824, 2909.492,
            262.2432, 425.5896, 504.1144, 605.604, 860.0688, 1068.604, 270.7624, 798.5824, 806.3608, 1024.5264,
            1168.9824, 1464.562, 1620.13, 581.528, 981.1896, 1277.139, 1819.0344, 2222.7704, 2000.5304, 913.4064,
            947.8536, 1228.2464, 1381.592, 2502.4224, 1517.8992, 1702.3584, 2070.9064, 2603.912,
            170.7544, 393.7352, 867.4768, 1116.756, 1318.624, 265.9472, 768.58, 729.688, 1009.7104, 1048.9728,
            1403.075, 1590.127, 375.956, 823.0288, 1104.162, 1662.3552, 2041.6448, 1793.8472, 653.756, 720.0576,
            977.1152, 1138.2392, 2336.4832, 1258.2488, 1469.0064, 1823.1088, 2347.2248,
            341.5088, 1024.8968, 1264.5456, 1457.524, 411.5144, 854.5128, 789.6928, 1094.9024, 1064.9, 1451.598,
            1655.318, 359.288, 815.9912, 1073.79, 1637.168, 1991.2704, 1720.508, 515.9672, 554.8592, 806.7312,
            967.8552, 2192.0272, 1114.5336, 1367.8872, 1700.5064, 2179.0632,
            986.7456, 1170.8344, 1326.032, 650.052, 1160.8336, 1113.7928, 1402.7048, 1404.9272, 1783.106, 1978.306,
            700.056, 1157.13, 1410.854, 1973.4912, 2315.3704, 2028.6808, 770.0616, 534.4872, 870.44, 981.56,
            1998.6784, 1320.1056, 1649.3912, 1942.3776, 2235.7344,
            281.1336, 511.8928, 789.6928, 1154.5368, 1226.7648, 1313.8088, 1606.7952, 1785.328, 1869.779, 1157.87,
            1484.193, 1781.994, 2280.5528, 2713.9208, 2535.388, 1518.64, 1503.824, 1811.9968, 1949.7856, 2908.7512,
            2122.7624, 2277.5896, 2662.4352, 3191.7368,
            231.8704, 1064.9, 1431.9664, 1507.528, 1582.3488, 1887.188, 2052.757, 2124.614, 1428.262, 1764.956,
            2063.128, 2560.5752, 2995.0544, 2815.4104, 1770.512, 1702.7288, 2026.4584, 2150.9128, 2998.388,
            2374.6344, 2550.5744, 2929.1232, 3402.4944,
            1287.14, 1663.8368, 1738.6576, 1812.7376, 2118.3176, 2282.405, 2349.077, 1644.576, 1992.752, 2291.294,
            2792.0752, 3225.4432, 3041.7248, 1969.7872, 1860.5192, 2193.5088, 2306.1104, 3057.2816, 2571.6872,
            2767.9992, 3139.14, 3560.6552,
            540.0432, 536.7096, 775.2472, 898.5904, 1200.837, 1367.146, 374.104, 720.428, 1018.97, 1553.0872,
            1963.8608, 1757.1776, 780.4328, 964.5216, 1185.28, 1359.368, 2599.4672, 1369.7392, 1487.8968, 1877.928,
            2528.7208,
            141.1224, 241.8712, 467.0744, 668.2016, 827.1032, 560.0448, 456.3328, 704.1304, 1135.6464, 1581.2376,
            1458.6352, 948.9648, 1335.6624, 1448.6344, 1639.7608, 3003.5736, 1404.1864, 1320.1056, 1750.5104,
            2653.5456,
            328.1744, 380.7712, 673.7576, 865.9952, 463.3704, 318.9144, 583.7504, 1053.788, 1489.7488, 1344.1816,
            823.7696, 1237.5064, 1327.884, 1520.1216, 2901.7136, 1263.8048, 1185.28, 1614.2032, 2514.6456,
            447.4432, 471.5192, 595.6032, 787.8408, 566.3416, 732.6512, 1050.8248, 1506.0464, 1441.2264, 1146.7584,
            1563.8288, 1652.7248, 1844.9624, 3229.5176, 1542.716, 1378.6288, 1811.9968, 2799.4832,
            440.776, 687.092, 706.3528, 269.6512, 290.764, 673.7576, 1114.904, 1003.784, 908.2208, 1403.816,
            1401.2232, 1589.3864, 3020.612, 1165.6488, 937.8528, 1369.7392, 2414.2672,
            251.5016, 1102.681, 704.8712, 665.6088, 697.0928, 1137.4984, 1171.2048, 1348.256, 1835.7024, 1841.9992,
            2029.792, 3461.388, 1573.4592, 1256.3968, 1666.4296, 2800.5944,
            1316.772, 945.2608, 916.3696, 868.9584, 1276.028, 1366.0352, 1589.3864, 2067.2024, 2087.5744, 2276.108,
            3704, 1824.22, 1492.712, 1892.744, 3046.54,
            458.5552, 730.0584, 1290.4736, 1665.6888, 1423.8176, 436.7016, 776.7288, 832.2888, 1094.1616, 2443.8992,
            1002.6728, 1123.4232, 1505.3056, 2194.2496,
            299.2832, 841.5488, 1243.4328, 1053.4176, 644.1256, 1134.1648, 1144.1656, 1334.1808, 2758.7392,
            976.3744, 866.736, 1295.2888, 2233.512,
            563.008, 946.0016, 761.9128, 778.5808, 1298.252, 1229.3576, 1408.2608, 2848.376, 907.8504, 647.4592,
            1080.0864, 2141.6528,
            455.2216, 504.8552, 1306.4008, 1824.9608, 1698.284, 1857.9264, 3278.4104, 1235.284, 761.5424, 1086.7536,
            2323.8896,
            372.9928, 1587.5344, 2082.0184, 1889.4104, 2020.9024, 3378.048, 1339.3664, 816.3616, 958.9656, 2220.548,
            1274.5464, 1747.5472, 1533.456, 1656.7992, 3005.0552, 970.448, 449.2952, 618.9384, 1883.484,
            520.7824, 507.0776, 699.3152, 2114.6136, 604.4928, 880.0704, 1186.7616, 1758.2888,
            346.324, 448.184, 1668.2816, 884.8856, 1318.2536, 1526.048, 1701.2472,
            192.2376, 1621.9816, 595.6032, 1086.7536, 1226.3944, 1379.74,
            1439.7448, 691.5368, 1207.504, 1287.8808, 1254.9152,
            2053.4976, 2569.8352, 2494.2736, 1592.3496,
            523.3752, 641.1624, 1257.1376,
            433.7384, 1574.5704,
            1265.6568]
citylist = [u'北京',	u'上海',	u'天津',	u'石家庄', u'太原', u'呼和浩特',
            u'沈阳',	u'长春',	u'哈尔滨', u'济南', u'南京', u'合肥',
            u'杭州', u'南昌', u'福州',u'台北', u'郑州', u'武汉', u'长沙',
            u'广州', u'海口', u'南宁', u'西安', u'银川', u'兰州',
            u'西宁', u'乌鲁木齐', u'成都', u'贵阳', u'昆明', u'拉萨']

distmatrix = generateDistMatrix(distlist)

MAX_ITER = 10000  #外循环(温度)迭代次数
MAX_M = 30   #内循环(热平衡)迭代次数
lambda_ = 0.96  #降温系数
T0 = 100  #初始温度
x0 = range(1,32)  #初始路径

T = T0
iter = 1
x = x0   #记录路径变量
xx = [x0]   #记录选择的所有路径
distances = [tsp_dist(distmatrix, x0)]  #路径对应的距离
n = 1   #路径更新计数

while iter <= MAX_ITER:   #外循环
    m = 1  #内循环迭代器
    while m <= MAX_M:   #内循环
        newx = tsp_new_path(x)
        old_dist = tsp_dist(distmatrix, x)
        new_dist = tsp_dist(distmatrix, newx)
        if old_dist > new_dist:
            x = newx
            xx.append(x)
            distances.append(new_dist)
            n += 1
        else:
            tmp = np.random.rand()
            if tmp < np.exp(-(new_dist-old_dist)/T):
                x = newx
                xx.append(x)
                distances.append(new_dist)
                n += 1
        m += 1   #内循环次数加1
    iter += 1   #外循环次数加1
    T = T*lambda_   #降温

route = xx[distances.index(min(distances))]
print min(distances)
for i in route:
    print citylist[i-1]
plt.plot(np.array(distances))
plt.ylabel("distances")
plt.xlabel("update times")
plt.show()
