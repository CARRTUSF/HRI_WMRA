/*
    Tian Tan 2/24/2019
    string subscriber
 */
using RosSharp.RosBridgeClient.Messages.Standard;

namespace RosSharp.RosBridgeClient
{
    public class StringSubscriber : Subscriber<String>
    {
        public String message;
        // Use this for initialization
        protected override void Start()
        {
            base.Start();
            message = new String(""); // empty data if no message received
        }

        // Update is called once per frame
        protected override void ReceiveMessage(String messageRecieved)
        {
            message = messageRecieved;
        }
    }

}

