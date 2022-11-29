#!/usr/bin/env
import rospy
from geometry_msgs.msg import PoseStamped

def talk(tuple1,tuple2):
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size= 10)
    rospy.init_node('publisher_node', anonymous=True)
    rospy.sleep(1)
    rospy.loginfo("Node position_node initialized. Desired position (x,y) = (%s, %s)" % (tuple1[0], tuple1[1]))


    #while not rospy.is_shutdown():
    goal = PoseStamped()

    goal.header.seq = 1
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"

    goal.pose.position.x = float(tuple1[0])
    goal.pose.position.y = float(tuple1[1])
    goal.pose.position.z = 0.0

    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = float(tuple2[0])
    goal.pose.orientation.w = float(tuple2[1])

    pub.publish(goal)



if __name__ == '__main__':

    try:

        while True:
            try:
                input_tuple = input("Enter desired position <x,y>:")
                tuple1 = tuple(input_tuple.split(","))

                input_tuple = input("Enter desired orientation <z,w>:")
                tuple2 = tuple(input_tuple.split(","))

                break
            except ValueError:
                print("Invalid entry.")

        talk(tuple1,tuple2)

    except rospy.ROSInterruptException:
        pass