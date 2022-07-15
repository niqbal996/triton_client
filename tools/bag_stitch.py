import rosbag
num_msgs = 500

with rosbag.Bag('test.bag', 'w') as outbag:
     for topic, msg, t in rosbag.Bag('/media/naeem/T7/Rosbags_Hacke_27.7.21/Field_test_d435_top_down_6_27_1627381316_0.bag').read_messages():
         while num_msgs:
             outbag.write(topic, msg, t)
             num_msgs -= 1