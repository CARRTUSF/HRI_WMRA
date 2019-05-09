
using UnityEngine;

namespace RosSharp.RosBridgeClient
{
    public class Vector3Subscriber : Subscriber<Messages.Geometry.Vector3>
    {
        public Vector3 received_message;

        protected override void Start()
        {
            base.Start();
            received_message = new Vector3(-1, -1, -1);
        }
        protected override void ReceiveMessage(Messages.Geometry.Vector3 message)
        {
            received_message.x = message.x;
            received_message.y = message.y;
            received_message.z = message.z;
        }
    }
}