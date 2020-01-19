//sep-20 2018, Tian Tan
//Receives ROS message of type Geometry_msgs/PoseArray

using RosSharp.RosBridgeClient.Messages.Geometry;

namespace RosSharp.RosBridgeClient
{
    public class PoseArraySubscriber : Subscriber<PoseArrayStamped>
    {
        public UnityEngine.Pose[] messageData;
        protected override void Start()
        {
            base.Start();           
        }
        protected override void ReceiveMessage(PoseArrayStamped message)
        {
            ProcessMessage(message);
        }

        private void ProcessMessage(PoseArrayStamped msg)
        {
            int n = msg.poses.Length;
            messageData = new UnityEngine.Pose[n];
            for(int i = 0; i<n; i++)
            {
                messageData[i].position.x = msg.poses[i].position.x;
                messageData[i].position.y = msg.poses[i].position.y;
                messageData[i].position.z = msg.poses[i].position.z;
                messageData[i].rotation.x = msg.poses[i].orientation.x;
                messageData[i].rotation.y = msg.poses[i].orientation.y;
                messageData[i].rotation.z = msg.poses[i].orientation.z;
                messageData[i].rotation.w = msg.poses[i].orientation.w;
            }
        }
    }
}