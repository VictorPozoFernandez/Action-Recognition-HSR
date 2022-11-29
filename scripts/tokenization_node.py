#!/usr/bin/env
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

def listen():
    rospy.init_node("tokenization_node", anonymous=True)
    rospy.loginfo("Node tokenization_node initialized. Listening...")
    rospy.Subscriber("/speech_recognition/final_result", String, callback)
    rospy.spin()

def speak(msg):
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size= 10)
    rospy.Rate(1)
    rospy.loginfo("Publishing...")

    if msg.data == "coffee":
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = float(2.3)
        goal.pose.position.y = float(4.1)
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = float(0.7)
        goal.pose.orientation.w = float(0.7)

        pub.publish(goal)
        rospy.Rate(1)
    
    elif msg.data == "humans":
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = float(0.6)
        goal.pose.position.y = float(3.45)
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = float(-1)
        goal.pose.orientation.w = float(0)

        pub.publish(goal)
        rospy.Rate(1)

    else:
        rospy.loginfo("Try again.")



def callback(msg):
    #rospy.loginfo("Received data: %s", msg.data)

    if msg.data == "coffee" or msg.data == "humans":
        speak(msg)




if __name__ == '__main__':
    try:
        listen()
    except rospy.ROSInterruptException:
        pass