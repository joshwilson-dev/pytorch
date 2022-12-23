# import torch.nn.functional as F
# import torch
# class_logits = torch.tensor([[ 1.0144, -0.6546, -1.1215,  4.3748, -2.8719, -1.2215],
#         [ 2.0700, -0.9571, -1.0576,  2.3516, -2.0132, -1.3225],
#         [ 2.2806, -1.7024, -1.3999,  3.9698, -2.6194, -1.6698],
#         [ 3.4964, -0.7877, -1.8423,  2.0264, -2.3022, -1.3555],
#         [ 1.2482, -1.5535, -1.1967,  3.8674, -2.3615, -0.5842],
#         [ 1.0808, -1.4326, -0.6148,  5.2759, -2.7678, -0.1647],
#         [ 1.4112, -1.0754, -1.7862,  4.7202, -2.7328, -0.2040],
#         [ 2.0857, -1.8069,  0.5203,  3.3450, -1.3567, -0.9820],
#         [ 0.4392,  0.0644, -2.6446,  4.9698, -1.6905, -0.5986],
#         [ 4.4892, -1.7189, -2.0548,  4.3668, -3.1236, -2.3478],
#         [ 0.5733, -1.0967, -1.2194,  2.9650, -1.7132,  1.8733],
#         [ 0.3003, -0.6900, -2.2978,  3.7158, -1.3883, -0.1139],
#         [ 3.6641, -2.5050,  0.6462,  1.6625, -1.3139, -1.2212],
#         [ 2.1609, -2.1128, -2.1761,  5.0656, -2.9982, -1.0094],
#         [ 1.3251, -2.3125, -0.5138, -1.0994, -1.9398,  3.8772]])
# filter = torch.tensor([1, 0, 0, 1, 0, 0]).nonzero().squeeze()
# class_logits_filtered = class_logits[:, filter]
# src = F.softmax(class_logits_filtered, -1)
# index = filter.repeat(len(class_logits), 1)
# scores = torch.zeros(len(class_logits), 6).scatter_(1, index, src)

# class Person:
#   def __init__(self, name, age):
#     self.name = name
#     self.age = age

#   def myfunc(self, word):
#     print("Hello my name is " + self.filter + word)

# def yourfunc(self, word):
#     print("Slim Shady " + self.filter + word)

# Person.myfunc = yourfunc
# p1 = Person("John", 36)
# p1.filter = "Mark"
# p1.myfunc("here")
# import json
# import os
# import torch
# model_path = os.path.join("../models/bird-detector-final")
# index_to_class = json.load(open(os.path.join(model_path, "index_to_class.json")))
# region = "Africa"
# birds_by_region = json.load(open(os.path.join(model_path, "birds_by_region.json")))
# regional_filter = torch.zeros(len(index_to_class) + 1)
# regional_filter[0] = 1
# for i in range(1, len(index_to_class) + 1):
#     species = ' '.join(index_to_class[str(i)].split("_")[-2:])
#     if species in birds_by_region[region]:
#         regional_filter[i] = 1

# print(regional_filter)


# output_dict = {"bird": [{"Chicken": 0.9}, {"Chook": 0.7}]}
# print(list(output_dict["bird"][0].values())[0])
# import torch
# scores = torch.tensor([[0.0117, 0.0104, 0.6808, 0.0063, 0.2285],
#         [0.0114, 0.0023, 0.9300, 0.0056, 0.0202],
#         [0.0017, 0.0407, 0.1125, 0.0057, 0.0063]])
# # print(torch.max(scores, 1).values)
# inds = torch.where(torch.max(scores, 1).values > 0.2)[0]
# print(inds)
# import torch
# import itertools
# import torchvision
# boxes = torch.tensor([[3327.7791, 1969.6235, 3385.2278, 2004.1595],
#         [2014.4736, 1500.4164, 2049.4221, 1551.6010],
#         [2302.1333, 1354.9839, 2360.5713, 1394.4214],
#         [2556.2854, 1524.4852, 2607.6470, 1552.8740],
#         [3406.5698, 1923.3138, 3460.7759, 1970.3207],
#         [3692.0271, 1827.6062, 3725.6228, 1893.7925],
#         [3175.5513, 2042.9064, 3216.6042, 2098.9644],
#         [3810.6763, 1372.8502, 3865.7637, 1421.3635],
#         [1896.0967, 1546.0492, 1936.5156, 1606.3052],
#         [2056.7236, 1462.4468, 2111.6389, 1495.2457],
#         [3250.3572, 2027.0790, 3299.6592, 2085.2231],
#         [2171.7085, 1419.5067, 2234.9465, 1452.6790],
#         [1833.9647, 1584.3945, 1864.6381, 1642.2498],
#         [3321.9480, 1396.9189, 3366.7976, 1452.9523],
#         [4250.0039, 1389.4354, 4321.2749, 1425.0665],
#         [3482.1824, 1899.2028, 3522.1531, 1939.7397],
#         [2705.2229, 1857.0608, 2770.7175, 1902.0510],
#         [2608.9302, 1394.2922, 2652.0737, 1444.8773],
#         [5244.8291, 2658.8240, 5313.8403, 2694.1946],
#         [2474.0925, 1340.4828, 2523.8140, 1411.5769],
#         [3141.7871, 1717.1541, 3189.5271, 1790.7892],
#         [2515.4922, 1405.5322, 2548.8223, 1462.4904],
#         [2603.2417, 1390.3702, 2651.4485, 1451.0399],
#         [2590.4468, 1600.6864, 2638.4641, 1675.6003],
#         [5243.2314, 2662.3640, 5318.1045, 2705.5227],
#         [3140.8313, 1715.8450, 3187.8044, 1788.8617],
#         [2473.5928, 1339.0201, 2525.2051, 1414.5156],
#         [2593.3186, 1600.5099, 2641.2971, 1678.3270],
#         [4252.3906, 1389.6050, 4315.4897, 1425.0973],
#         [3141.8862, 1717.6533, 3186.3564, 1788.3265],
#         [4247.9131, 1389.5306, 4318.0684, 1426.6135],
#         [2550.8228, 1518.7617, 2611.2476, 1561.9939],
#         [1826.7925, 1587.3184, 1864.8732, 1641.5145],
#         [1821.8468, 1585.3936, 1874.5900, 1645.1609],
#         [3165.6311, 2041.3822, 3222.3975, 2104.7964],
#         [3476.2705, 1890.8848, 3526.2874, 1949.6104],
#         [1827.7576, 1584.0582, 1871.5582, 1646.0233],
#         [4254.0498, 1389.4404, 4321.1860, 1424.9778],
#         [2606.9631, 1400.8330, 2651.7407, 1448.2389],
#         [2556.3059, 1519.6365, 2605.8655, 1559.5229],
#         [3252.7432, 2032.6669, 3296.0989, 2083.8838],
#         [3804.7993, 1376.2842, 3867.7383, 1419.2526],
#         [2174.5542, 1418.2987, 2229.0166, 1459.8921],
#         [3135.4583, 1720.5524, 3184.4033, 1786.2589],
#         [2054.3081, 1461.9711, 2114.2874, 1508.6600],
#         [2599.3936, 1389.2982, 2653.9763, 1456.1908],
#         [2474.3691, 1343.5626, 2521.3538, 1409.2256],
#         [3169.9548, 2040.8333, 3220.6750, 2106.9321],
#         [3470.4167, 1886.7102, 3527.3625, 1943.5542],
#         [2169.3201, 1415.8638, 2236.0247, 1456.5222],
#         [3136.5427, 1726.6737, 3185.5674, 1782.8938],
#         [3684.0938, 1826.7285, 3735.8352, 1889.6497],
#         [4258.9082, 1391.7870, 4314.6733, 1426.9410],
#         [2298.0244, 1355.3442, 2367.3481, 1405.4034],
#         [1823.4655, 1584.0494, 1869.5034, 1643.6389],
#         [3247.0742, 2026.3994, 3299.2629, 2088.6414],
#         [2469.6819, 1344.6635, 2519.4121, 1410.8130],
#         [2603.1558, 1394.4553, 2650.5327, 1450.8427],
#         [2592.2451, 1599.6151, 2638.5439, 1674.6118],
#         [3805.8267, 1378.5013, 3863.9961, 1418.5781],
#         [1893.0942, 1549.2181, 1932.2356, 1606.4520],
#         [2053.0967, 1459.4141, 2114.1079, 1508.5007],
#         [2585.4827, 1606.2772, 2636.4075, 1670.1698],
#         [2553.2080, 1517.2123, 2607.1428, 1558.5396],
#         [3168.4363, 2051.9373, 3217.1082, 2101.7183],
#         [5249.4741, 2659.4712, 5310.6270, 2698.8765],
#         [3166.0635, 2046.9218, 3219.5203, 2102.2051],
#         [3683.7886, 1825.5377, 3733.0469, 1893.9272],
#         [3479.1907, 1894.8569, 3521.3237, 1945.1621],
#         [3405.4285, 1922.3402, 3458.8823, 1977.4761],
#         [2012.2058, 1495.1057, 2051.3835, 1555.1350],
#         [2473.6221, 1353.1332, 2520.7144, 1407.0522],
#         [2056.6006, 1461.3636, 2109.2776, 1503.0768],
#         [3808.9585, 1374.2095, 3859.3418, 1422.8295],
#         [2166.5713, 1413.0024, 2228.3262, 1464.8105],
#         [3248.3167, 2027.6331, 3296.2930, 2086.3689],
#         [1891.8391, 1547.3068, 1936.5813, 1607.9634],
#         [3474.3303, 1892.3595, 3524.1255, 1942.6215],
#         [2301.8110, 1356.0635, 2357.8513, 1398.1504],
#         [2010.9015, 1496.9991, 2051.9133, 1555.5436],
#         [2502.7197, 1414.0394, 2559.0481, 1464.1838],
#         [3328.0369, 1962.9731, 3388.3909, 2012.8367],
#         [2170.4158, 1412.9945, 2227.7100, 1464.8612],
#         [3246.9810, 2027.0713, 3299.3159, 2089.7986],
#         [2548.3357, 1516.2091, 2609.6643, 1560.8539],
#         [3322.3132, 1397.2048, 3369.5820, 1461.5017],
#         [3407.1423, 1923.4922, 3456.0056, 1971.9666],
#         [3684.2742, 1838.1503, 3727.4072, 1885.3427],
#         [3806.4097, 1381.2321, 3859.5928, 1422.9716],
#         [2055.7837, 1459.2821, 2112.2393, 1505.0594],
#         [2587.7166, 1606.5477, 2637.7822, 1669.3564],
#         [3400.4634, 1919.5470, 3457.5923, 1973.6257],
#         [2506.4270, 1412.4163, 2555.6707, 1467.5083],
#         [3403.2317, 1923.1252, 3456.1248, 1972.5607],
#         [2691.1206, 1860.7145, 2768.4707, 1900.5375],
#         [2294.1709, 1353.1910, 2363.6838, 1403.3234],
#         [5243.1455, 2659.5435, 5313.1001, 2703.0400],
#         [3682.8357, 1832.1863, 3729.0435, 1888.3810],
#         [2715.1292, 1858.5664, 2756.6951, 1904.8882],
#         [3320.7817, 1964.8582, 3385.6267, 2013.8102],
#         [2016.8740, 1498.8679, 2047.0657, 1556.6270],
#         [2705.0320, 1859.3076, 2769.2297, 1905.2969],
#         [3326.6250, 1966.6628, 3380.6072, 2013.7321],
#         [2518.2646, 1413.5778, 2546.1040, 1463.5614],
#         [3328.6951, 1400.3223, 3358.2200, 1457.2113],
#         [1887.4023, 1547.1083, 1934.2231, 1608.6886],
#         [3320.8406, 1397.6757, 3369.2478, 1454.8496],
#         [5243.4883, 2659.8916, 5312.2407, 2697.5876],
#         [2298.9460, 1353.9331, 2363.5208, 1402.7480],
#         [3323.4583, 1964.4655, 3381.5715, 2013.7997],
#         [2505.6426, 1410.2272, 2550.6038, 1463.9279],
#         [2013.5890, 1496.5607, 2052.2502, 1553.0977],
#         [3321.7639, 1395.4548, 3363.9377, 1454.8322],
#         [2698.2805, 1859.8571, 2764.9231, 1901.7992]])

# labels = ['Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Australian Wood Duck_aves_adult_anseriformes_anatidae_chenonetta_jubata', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Masked Lapwing_aves_adult_charadriiformes_charadriidae_vanellus_miles', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Pied Stilt_aves_adult_charadriiformes_recurvirostridae_himantopus_leucocephalus', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Australian White Ibis_aves_adult_pelecaniformes_threskiornithidae_threskiornis_molucca', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae', 'Silver Gull_aves_adult_charadriiformes_laridae_chroicocephalus_novaehollandiae']
# scores = [0.9933859705924988, 0.9890907406806946, 0.9824638962745667, 0.982067883014679, 0.9802435636520386, 0.9789391756057739, 0.9787594079971313, 0.9745098352432251, 0.9718745946884155, 0.9672689437866211, 0.9592401385307312, 0.9517217874526978, 0.9498605728149414, 0.9444117546081543, 0.9392508268356323, 0.9129443168640137, 0.8791624903678894, 0.7855184674263, 0.7820486426353455, 0.7292629480361938, 0.5234526991844177, 0.4681711792945862, 0.2285144329071045, 0.1125466451048851, 0.05611862614750862, 0.04706158488988876, 0.043264247477054596, 0.04073507338762283, 0.028053514659404755, 0.02436203509569168, 0.022687673568725586, 0.020196247845888138, 0.01963355951011181, 0.01775592938065529, 0.017245955765247345, 0.015221145935356617, 0.013624371029436588, 0.012937196530401707, 0.011723587289452553, 0.011351579800248146, 0.011281385086476803, 0.0112399160861969, 0.010946838185191154, 0.010838947258889675, 0.0106264753267169, 0.010369221679866314, 0.009631725028157234, 0.00902459118515253, 0.008973325602710247, 0.008971607312560081, 0.008489744737744331, 0.008428922854363918, 0.0071710869669914246, 0.006929297465831041, 0.006828753277659416, 0.006686358246952295, 0.006621542386710644, 0.0063284155912697315, 0.006294879596680403, 0.006257706321775913, 0.006232223939150572, 0.005759637802839279, 0.005737224128097296, 0.005647068843245506, 0.005605502985417843, 0.00498961890116334, 0.004792718682438135, 0.004578513093292713, 0.0044791363179683685, 0.0044725858606398106, 0.004236447159200907, 0.004221358336508274, 0.004031243734061718, 0.0038513308390975, 0.00381318642757833, 0.0036854019854217768, 0.003535417839884758, 0.0032626274041831493, 0.002899028593674302, 0.0027009372133761644, 0.002691440051421523, 0.002583903493359685, 0.0024073629174381495, 0.0023895257618278265, 0.002274031052365899, 0.0021718621719628572, 0.0020562936551868916, 0.0019010828109458089, 0.0018603341886773705, 0.0017969858599826694, 0.0017435778863728046, 0.0017377916956320405, 0.0016795756528154016, 0.0015404600417241454, 0.0014989475021138787, 0.0014241532189771533, 0.0013804819900542498, 0.0013513853773474693, 0.0012589163379743695, 0.0012095943093299866, 0.0011921549448743463, 0.0011918668169528246, 0.0011310299159958959, 0.001065222779288888, 0.0010224641300737858, 0.0006787384045310318, 0.0006763056153431535, 0.0005859488737769425, 0.0005526796448975801, 0.0005062651471234858, 0.0004713438975159079, 0.0003136754676233977, 0.00029722857289016247, 0.00021237062173895538]
# # find iou's that overlap
# output_dict = {"boxes": [], "scores": [], "labels": []}
# ious = torchvision.ops.box_iou(boxes, boxes[0].unsqueeze(0))
# for _ in range(len(ious)):

#     inds = (ious > 0.3).nonzero()[:,0].tolist()
#     print(inds)
#     # get corresponding boxes, scores and labels
#     bxs = [boxes[i] for i in inds]
#     scrs = [scores[i] for i in inds]
#     lbs = [labels[i] for i in inds]
#     # add to new group
#     output_dict["boxes"].append(bxs)
#     output_dict["scores"].append(scrs)
#     output_dict["labels"].append(lbs)
#     # remove used indices
#     for i in inds:
#         ious = torch.cat([ious[0:i], ious[i+1:]])
#         boxes = torch.cat([boxes[0:i], boxes[i+1:]])
#         del scores[i]
#         del labels[i]
# print(boxes)
# scores = 
# labels = 
# res = ious.clone()
# res[ious<=0.7] = 0
# res[ious>0.7] = 1
# print(res)
# print(boxes)
# print(boxes[res])
# boxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# inds = torch.tensor([[0, 2], [0, 0], [1]])
# print(boxes[inds[0]])
# filtered_boxes = torch.gather(boxes, 1, inds)
# print(filtered_boxes)
# print(tensor1)
# print(tensor2[2])
# print(tensor2[2]*tensor1)
# print(torch.matmul(tensor2[0], tensor1))
# print(torch.matmul(tensor2[1], tensor1))
# print(torch.matmul(tensor2[2], tensor1))
# boxes = torch.tensor([[3327.7791, 1969.6235, 3385.2278, 2004.1595],
#         [2014.4736, 1500.4164, 2049.4221, 1551.6010],
#         [2302.1333, 1354.9839, 2360.5713, 1394.4214]])
# ious = torch.Tensor([[0.6, 0.71, 1], [0, 0.4, 0.9], [1, 0.2, 0.3]])
# keep = (ious > 0.7).nonzero()
# filtered_boxes = boxes[keep]
# print(filtered_boxes)

import torch
import math
scores = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
trial = scores.repeat_interleave(5, dim = 0)

print(trial)