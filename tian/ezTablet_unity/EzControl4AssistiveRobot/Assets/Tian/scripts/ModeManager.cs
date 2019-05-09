
using UnityEngine;
using UnityEngine.UI;

namespace RosSharp.RosBridgeClient
{
    public class ModeManager : MonoBehaviour
    {
        public GameObject ROSFeedbackSubscriber;
        private Vector3Subscriber Vec3Sub;
        public float mode;
        private Vector3 ROS_feedback_message;
        public UnityEngine.UI.Button btn;
        private float oldmessage_index;

        public void Start()
        {
            mode = 1;
            oldmessage_index = -1;
            btn.GetComponentInChildren<Text>().text = "Manual";
            Vec3Sub = ROSFeedbackSubscriber.GetComponent<Vector3Subscriber>();
        }
        public void Update()
        {
            ROS_feedback_message = Vec3Sub.received_message;
            if (ROS_feedback_message.x == 2 && ROS_feedback_message.y == 1 && ROS_feedback_message.z != oldmessage_index)
            {
                //Debug.Log("this should not be printed multiple times");
                mode = 1;//after reset robot position reset mode to 1
                oldmessage_index = ROS_feedback_message.z;
            }
        }
        public void changemode()
        {
            if (mode == 1)
            {
                mode = 2;
                btn.GetComponentInChildren<Text>().text = "Auto";
            }
            else
            {
                mode = 1;
                btn.GetComponentInChildren<Text>().text = "Manual";
            }
        }
        public void changetomode3()
        {
            mode = 3;

        }
        public void changetomode4()
        {
            if (mode != 4)
            {
                mode = 4;
            }
            else
            {
                mode = 1;
            }
            
        }
    }

}
