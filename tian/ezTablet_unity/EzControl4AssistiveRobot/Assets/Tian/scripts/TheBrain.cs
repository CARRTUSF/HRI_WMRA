using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;


namespace RosSharp.RosBridgeClient
{
    public class TheBrain : MonoBehaviour
    {

        public Vector2 user_input;
        public UIfunctions UIcontrol;
        public GameObject ros_connector;
        //public ObjectDataBase obj_DB;

        private Quaternion grasp_orientation;
        private PoseArraySubscriber PAS;
        private StringSubscriber IDS;
        public Pose[] pose_array;
        public string[] object_ids;
        public Vector2[] obj_screen_location;
        public int n_objects;
        // camera parameters
        private float focal_length = 525f;
        // display object location
        public GameObject indicator_prefab;
        private GameObject[] indicators;
        public bool indicators_ON = false;
        //public Camera mainCamera;

        // user gameobject interaction
        private RaycastHit hit;
        public int WhichGripper;

        // feedback ui
        private Vector3Subscriber FDS;
        public Vector3 feedback_message;

        void Start()
        {
            PAS = ros_connector.GetComponent<PoseArraySubscriber>();
            IDS = ros_connector.GetComponent<StringSubscriber>();
            FDS = ros_connector.GetComponent<Vector3Subscriber>();

            indicators = new GameObject[20];
            for (int i = 0; i < 20; i++)
            {
                indicators[i] = Instantiate(indicator_prefab) as GameObject;
                indicators[i].name = indicators[i].GetInstanceID().ToString();
                indicators[i].SetActive(false);
            }

        }

        private void RecognizedObjectUpdate()
        {
            n_objects = PAS.messageData.Length;
            //Debug.Log("number of objects: " + n_objects);
            pose_array = new Pose[n_objects];
            pose_array= PAS.messageData;
            object_ids = IDS.message.data.Split(' ');
            Object_on_screen_position(pose_array);
        }
        private void FeedbackUpdate()
        {
            /*
                (1,0,count) found solution ask for permission for execution
                (0,0,count) no solution found ask for instructions 
                (1,1,count) execution finished reset UI
                (1,2,count) execution failed ask for instructions
                other hide
             */            
            if (feedback_message.z != FDS.received_message.z)
            {
                Debug.Log("new feedback message recieved : " + FDS.received_message.z);
                feedback_message = FDS.received_message;
                Debug.Log(feedback_message);
                if (feedback_message.x == 1)
                {
                    if (feedback_message.y == 0)
                    {
                        Debug.Log("found solution, execute?");                      
                        UIcontrol.FeedbackUI(new Vector2(1,0));

                    }
                    else if (feedback_message.y == 1)
                    { 
                        Debug.Log("execution finished");
                        UIcontrol.FeedbackUI(new Vector2(1, 1));
                    }
                    else
                    {
                        Debug.Log("execution failed");
                        UIcontrol.FeedbackUI(new Vector2(1, feedback_message.y));
                    }
                }
                else if (feedback_message.x==0 && feedback_message.y==0)
                {
                    Debug.Log("no solution found, what to do next?");
                    UIcontrol.FeedbackUI(new Vector2(0, 0));
                }
            }
        }

        private void Object_on_screen_position(Pose[] poses)
        {
            obj_screen_location = new Vector2[n_objects];           
            for (int i = 0; i < n_objects; i++)
            {              
                Vector3 positionInKinect = poses[i].position;
                Vector3 normalized_position; // on image position
                normalized_position.x = (focal_length * (positionInKinect.x / positionInKinect.z) + 320) / 640;
                normalized_position.y = (240 - focal_length * (positionInKinect.y / positionInKinect.z)) / 480;
                normalized_position.z =  2f;
                //Debug.Log("norm_position:   " + normalized_position);
                obj_screen_location[i] = Camera.main.ViewportToScreenPoint(normalized_position);
            }           
        }

        public void Show_all_objects()
        {
            Debug.Log("showing all objects/////////////");
            for (int i = 0; i < n_objects; i++)
            {
                Vector3 position_indicator = indicators[i].transform.position;
                position_indicator.x = obj_screen_location[i].x;
                position_indicator.y = obj_screen_location[i].y;
                position_indicator.z = 5f;
                indicators[i].transform.position = Camera.main.ScreenToWorldPoint(position_indicator);
                indicators[i].SetActive(true);
            }  
        }

        private void Turnoff_indicators()
        {
            for (int i = 0; i < 20; i++)
            {
                indicators[i].SetActive(false);
            }
        }

        private bool IsPointerOverUIObject()
        {
            PointerEventData eventDataCurrentPosition = new PointerEventData(EventSystem.current);
            eventDataCurrentPosition.position = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
            List<RaycastResult> results = new List<RaycastResult>();
            EventSystem.current.RaycastAll(eventDataCurrentPosition, results);
            return results.Count > 0;
        }


        private bool IsTouchOverUIObject(int index)
        {
            PointerEventData eventDataCurrentPosition = new PointerEventData(EventSystem.current);
            eventDataCurrentPosition.position = new Vector2(Input.GetTouch(index).position.x, Input.GetTouch(index).position.y);
            List<RaycastResult> results = new List<RaycastResult>();
            EventSystem.current.RaycastAll(eventDataCurrentPosition, results);
            return results.Count > 0;
        }

        public void Change_grasp_orientation(float rot_angle_x)
        {
            Quaternion rot_q = Quaternion.Euler(rot_angle_x, 0f, 0f);
            grasp_orientation = grasp_orientation * rot_q;
        }
        // Update is called once per frame
        void Update()
        {
            RecognizedObjectUpdate();
            FeedbackUpdate();

#if UNITY_EDITOR
            if (Input.GetMouseButtonDown(0))    //-------------------------------------------user input happen
            {
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))      //-----------------------------------cast screen point input to a ray in gameworld              
                {
                    if (!IsPointerOverUIObject())       //-----------------------------------check if user is interacting with UI
                    {
                        GameObject recipient = hit.transform.gameObject;
                        Debug.Log("ray hit**************"+recipient.name);
                        if (recipient.name == "Gripper1")
                        {
                            Debug.Log("add waht happens when gripper 1 selected");
                            WhichGripper = 1;
                            //UIcontrol.GameObject_activation_switch("gripper2","deactivate");  
                            UIcontrol.HighLightGripper("Gripper1");
                            UIcontrol.ShowDefaultApply();
                            UIcontrol.Goal_pose_update("Gripper1");
                        }
                        else if (recipient.name == "Gripper2")
                        {
                            Debug.Log("add waht happens when gripper 2 selected");
                            WhichGripper = 2;
                            //UIcontrol.GameObject_activation_switch("gripper1", "deactivate");
                            UIcontrol.HighLightGripper("Gripper2");
                            UIcontrol.ShowDefaultApply();
                            UIcontrol.Goal_pose_update("Gripper2");
                        }
                        else if (recipient.name == "ImgPlane" && UIcontrol.default_grasp_UI.gameObject.activeSelf == false)
                        {
                            Debug.Log("add waht happens when background was hit");                           
                            user_input.x = Camera.main.ScreenToViewportPoint(Input.mousePosition).x;
                            user_input.y = 1 - Camera.main.ScreenToViewportPoint(Input.mousePosition).y;
                            Debug.Log("normalized clicked: " + user_input);
                            // user selection menu control-------------                    
                            UIcontrol.Toggle_user_menu(Input.mousePosition, new Vector2(user_input.x, 1 - user_input.y));
                        }

                    }
                }
            }
#endif
            // -----------------------------------------------touch screen input-------------------------------------------------
            for (int i = 0; i < Input.touchCount; ++i) 
            {
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))      //-----------------------------------cast screen point input to a ray in gameworld              
                {
                    if (!IsTouchOverUIObject(i))       //-----------------------------------check if user is interacting with UI
                    {
                        GameObject recipient = hit.transform.gameObject;
                        Debug.Log("ray hit**************" + recipient.name);
                        if (recipient.name == "Gripper1")
                        {
                            Debug.Log("add waht happens when gripper 1 selected");
                            WhichGripper = 1;  
                            UIcontrol.HighLightGripper("Gripper1");
                            UIcontrol.ShowDefaultApply();
                            UIcontrol.Goal_pose_update("Gripper1");
                        }
                        else if (recipient.name == "Gripper2")
                        {
                            Debug.Log("add waht happens when gripper 2 selected");
                            WhichGripper = 2;
                            UIcontrol.HighLightGripper("Gripper2");
                            UIcontrol.ShowDefaultApply();
                            UIcontrol.Goal_pose_update("Gripper2");
                        }
                        else if (recipient.name == "ImgPlane" && UIcontrol.default_grasp_UI.gameObject.activeSelf == false)
                        {
                            if (Input.GetTouch(i).phase == TouchPhase.Ended)
                            {
                                user_input.x = Camera.main.ScreenToViewportPoint(Input.GetTouch(i).position).x;
                                user_input.y = 1 - Camera.main.ScreenToViewportPoint(Input.GetTouch(i).position).y;
                                // user selection menu control-------------
                                UIcontrol.Toggle_user_menu(Input.GetTouch(i).position, new Vector2(user_input.x, 1 - user_input.y));
                            }
                        }

                    }
                }
            }

        }
    }


}
