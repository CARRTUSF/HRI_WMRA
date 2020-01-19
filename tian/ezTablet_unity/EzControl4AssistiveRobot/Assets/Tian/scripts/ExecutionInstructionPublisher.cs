/*
    tian tan 2/28/2019
 */

using UnityEngine;

namespace RosSharp.RosBridgeClient
{
    public class ExecutionInstructionPublisher : Publisher<Messages.Geometry.Vector3>
    {
        public UIfunctions execution_instruction;
        private Messages.Geometry.Vector3 message;

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
            message = new Messages.Geometry.Vector3();
        }

        private void UpdateMessage()
        {
            // check if is new instruction
            message.x = execution_instruction.mode;
            message.y = execution_instruction.attempt;
            message.z = execution_instruction.confirmation;           
            Debug.Log("Publishing EI: \n" + message.x +"  " + message.y + "  " + message.z);
            Publish(message);
        }
    }
}
