#!/usr/bin/env python
import rospy
import yaml
import tf2_ros
import geometry_msgs.msg
import os

def load_transformation_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    transform = {}
    transform['translation'] = (
        data['transformation']['x'],
        data['transformation']['y'],
        data['transformation']['z']
    )
    transform['rotation'] = (
        data['transformation']['qx'],
        data['transformation']['qy'],
        data['transformation']['qz'],
        data['transformation']['qw']
    )
    transform['parent_frame'] = data['parameters']['robot_base_frame']
    transform['child_frame'] = data['parameters']['tracking_base_frame']
    return transform

def publish_static_tf(transform):
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_tf = geometry_msgs.msg.TransformStamped()

    static_tf.header.stamp = rospy.Time.now()
    static_tf.header.frame_id = transform['parent_frame']
    static_tf.child_frame_id = transform['child_frame']

    static_tf.transform.translation.x = transform['translation'][0]
    static_tf.transform.translation.y = transform['translation'][1]
    static_tf.transform.translation.z = transform['translation'][2]

    static_tf.transform.rotation.x = transform['rotation'][0]
    static_tf.transform.rotation.y = transform['rotation'][1]
    static_tf.transform.rotation.z = transform['rotation'][2]
    static_tf.transform.rotation.w = transform['rotation'][3]

    rospy.loginfo("Publishing static TF from %s to %s", static_tf.header.frame_id, static_tf.child_frame_id)
    static_broadcaster.sendTransform(static_tf)

if __name__ == '__main__':
    rospy.init_node('handeye_static_tf_publisher')

    # 原本直接使用當前工作目錄下的 handeye_calibration.yaml
    # yaml_file = os.path.join(os.getcwd(), "handeye_calibration.yaml")
    # if not os.path.exists(yaml_file):
    #     rospy.logerr("YAML file not found at: %s", yaml_file)
    #     exit(1)

    # 改為從參數 ~yaml_file 讀取，若未提供則退回當前工作目錄
    yaml_file = rospy.get_param('~yaml_file', os.path.join(os.getcwd(), "handeye_calibration.yaml"))
    if not os.path.exists(yaml_file):
        rospy.logerr("YAML file not found at: %s", yaml_file)
        exit(1)

    transform = load_transformation_from_yaml(yaml_file)
    publish_static_tf(transform)
    rospy.spin()  # Keep alive
