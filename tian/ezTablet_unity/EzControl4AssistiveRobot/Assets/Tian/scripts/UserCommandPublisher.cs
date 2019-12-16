/*
    tian tan 2/28/2019
 */
using UnityEngine;


namespace RosSharp.RosBridgeClient
{
    public class UserCommandPublisher : Publisher<Messages.Geometry.PoseStamped>
    {
        public UIfunctions user_input;
        private Messages.Geometry.PoseStamped message;
        public string FrameId = "Unity";
        
        protected override void Start()
        {
            base.Start();
            InitializeMessage();
        }

        private void Update()
        {
            UpdateMessage();
        }

        private void InitializeMessage()
        {
            message = new Messages.Geometry.PoseStamped
            {
                header = new Messages.Standard.Header()
                {
                    frame_id = FrameId
                }
            };
        }

        private void UpdateMessage()
        {
            message.header.Update();
            message.pose.position.x = user_input.goal_pose.position.x;
            message.pose.position.y = user_input.goal_pose.position.y;
            message.pose.position.z = user_input.goal_pose.position.z;
            message.pose.orientation.x = user_input.goal_pose.rotation.x;
            message.pose.orientation.y = user_input.goal_pose.rotation.y;
            message.pose.orientation.z = user_input.goal_pose.rotation.z;
            message.pose.orientation.w = user_input.goal_pose.rotation.w;
            Debug.Log("user command : \n " + message.pose.position.x + "," + message.pose.position.y + "," + message.pose.position.z + ", (" + message.pose.orientation.x + ","
                + message.pose.orientation.y + "," + message.pose.orientation.z + ","+ message.pose.orientation.w + ")");
            Publish(message);
        }

    }

}

