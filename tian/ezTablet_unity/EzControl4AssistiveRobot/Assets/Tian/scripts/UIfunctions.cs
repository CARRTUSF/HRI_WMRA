/*
 Tian Tan - 2/17/2019

 */
using UnityEngine;
using UnityEngine.SceneManagement;
//using UnityEngine.UI;
using RosSharp.RosBridgeClient;

public class UIfunctions : MonoBehaviour
{
    // IP input----
    public static string IP;
    private UnityEngine.UI.InputField Input;
    // user menu control ---
    private GameObject user_menu;
    private bool is_user_menu_open = false;
    // user menu button on click------
    private UnityEngine.UI.Button btnGrasp;
    // Gripper UI
    private GameObject gripper1;
    private GameObject gripper2;
    private float z_angle;
    private float x_angle;
    private float dz;               //initial grasp-o-z-q and dz are same for all grasps
    private ObjectDataBase obj_DB;
    public GameObject default_grasp_UI;
    private GameObject grasp_modifier;
    private Pose object_in_camera; // object orientation in kinect/other camera in real world the inverse quaternion is the object in unity
    private Vector3 object_in_unity_position;
    // user input
    private GameObject manager;
    private TheBrain mainscript;
    private string object_of_interest;
    public int object_in_array;

    // user input publishing
    public float mode;
    public float attempt;
    public float confirmation;
    public Pose goal_pose; //robot pose control cmd
    // feedback ui
    private GameObject feedback_menu;
    private UnityEngine.UI.Text msgfeedback;
    void Awake()
    {
        Debug.Log("started");
        string CurrentScene = SceneManager.GetActiveScene().name;
        if(CurrentScene == "start")
        {
            Debug.Log("scene 0");
            Input = GameObject.Find("Canvas").GetComponentInChildren<UnityEngine.UI.InputField>();
        }
        else
        {
            Debug.Log("scene 1"); 
            // user menu control
            user_menu = GameObject.Find("userMenu");
            user_menu.gameObject.SetActive(false);
 
            btnGrasp = user_menu.GetComponentsInChildren<UnityEngine.UI.Button>()[0];
            btnGrasp.onClick.AddListener(BtnGraspclick);
            btnGrasp.GetComponentInChildren<UnityEngine.UI.Text>().text = "Show Recognized";
            // Gripper UI
            gripper1 = GameObject.Find("Gripper1");
            gripper2 = GameObject.Find("Gripper2");
            gripper1.SetActive(false);
            gripper2.SetActive(false);
            obj_DB = GetComponent<ObjectDataBase>();
            // grasp modifier control
            grasp_modifier = GameObject.Find("GraspModifier");
            default_grasp_UI = GameObject.Find("OKnMore");
            grasp_modifier.SetActive(false);
            default_grasp_UI.SetActive(false);

            // user input
            manager = GameObject.Find("TheBoss");
            mainscript = manager.GetComponent<TheBrain>();
            // feedback
            feedback_menu = GameObject.Find("feedbackMenu");
            feedback_menu.SetActive(false);
            msgfeedback = feedback_menu.transform.Find("FeedbackMsg").GetComponentInChildren<UnityEngine.UI.Text>();
        }
    }

    //---------------------------------------- button functions----------------------------
    public void Launch()
    {
        SceneManager.LoadScene(1);
    }

    public void Quit()
    {
        Debug.Log("has quit");
        Application.Quit();
    }

    public void BtnResetClick()
    {
        Debug.Log("requesting ROS motion planner to find solution for reset arm/////////////////\\\\\\\\\\\\\\\\\\");
        feedback_menu.SetActive(true);
        mode = 3;
        attempt += 1;
    }

    public void BtnModeClick()
    {
        if (mode != 4)
        {
            Debug.Log("switch to manual/////////////////\\\\\\\\\\\\\\\\\\");
            mode = 4;
            attempt += 1;
        }
        else
        {
            Debug.Log("switch to auto/////////////////\\\\\\\\\\\\\\\\\\");
            mode = 1;
            attempt += 1;
        }
    }

    public void BtnApplyClick()
    {
        feedback_menu.SetActive(true);
        mode = 1;
        attempt += 1;
    }

    public void BtnConfirmClick()
    {
        confirmation = 1;
        msgfeedback.text = "Waiting for execution...";
    }

    public void BtnCancelClick()
    {
        grasp_modifier.SetActive(false);
        gripper2.SetActive(false);
        gripper1.SetActive(false);
        default_grasp_UI.SetActive(false);
        is_user_menu_open = false;
        feedback_menu.SetActive(false);
    }

    public void BtnMoreClick()
    {
        gripper1.SetActive(false);
        HighLightGripper("Gripper2");
        default_grasp_UI.SetActive(false);
        grasp_modifier.SetActive(true);
        Initialize_grasp_tracking();
    }

    public void BtnMoveClick()
    {
        user_menu.SetActive(false);
        is_user_menu_open = false;
        feedback_menu.SetActive(true);
        goal_pose.position.x = mainscript.user_input.x;
        goal_pose.position.y = mainscript.user_input.y;
        Debug.Log("move clicked");
        mode = 2;
        attempt += 1;
    }

    public void BtnGraspclick()
    {
        if (btnGrasp.GetComponentInChildren<UnityEngine.UI.Text>().text == "Grasp")
        {
            Debug.Log("grasp clicked");
            Quaternion grasp1pose = SetGripperOrientation(obj_DB.object_data_base[object_of_interest][0]);
            Quaternion grasp2pose = SetGripperOrientation(obj_DB.object_data_base[object_of_interest][1]);

            Vector3 pre_grasp1_p = SetPreGraspPosition(grasp1pose, object_in_unity_position);
            Vector3 pre_grasp2_p = SetPreGraspPosition(grasp2pose, object_in_unity_position);

            gripper1.transform.position = pre_grasp1_p;
            gripper2.transform.position = pre_grasp2_p;
            gripper1.transform.rotation = grasp1pose;
            gripper2.transform.rotation = grasp2pose;

            gripper1.SetActive(true);
            gripper2.SetActive(true);
            HighLightGripper("none");
            user_menu.SetActive(false);
            default_grasp_UI.SetActive(true);
            default_grasp_UI.transform.Find("btnApply2").gameObject.SetActive(false);
        }
        else
        {
            Debug.Log("show all objects+++++++++++++");
            mainscript.Show_all_objects();
            is_user_menu_open = false;
            user_menu.SetActive(false);
        }
    }
    //--------------------------------------------------------button functions-------------------

 
    //===================================================slider functions=====================================
    public void Slider_orientation_z_control(float slidervalue)
    {
        z_angle = -45 - slidervalue * 180;
        Quaternion grasp_o_z_q = Quaternion.Euler(new Vector3(0f, 0f, z_angle));
        Quaternion grasp_o_x_q = Quaternion.Euler(new Vector3(x_angle, 0f, 0f));
        Quaternion girrper2_new_q = InverseQuaternion(object_in_camera.rotation) * grasp_o_z_q * grasp_o_x_q;
        gripper2.transform.rotation = girrper2_new_q;
        gripper2.transform.position = SetPreGraspPosition(girrper2_new_q, object_in_unity_position, dz);       
        // set up goal pose
        Goal_pose_update("custom");
    }

    public void Slider_orientation_x_control(float slidervalue)
    {
        x_angle = 45 * (slidervalue - 1) + 180;
        Quaternion grasp_o_x_q = Quaternion.Euler(new Vector3(x_angle, 0f, 0f));
        Quaternion grasp_o_z_q = Quaternion.Euler(new Vector3(0f, 0f, z_angle));
        Quaternion gripper2_new_q = InverseQuaternion(object_in_camera.rotation) * grasp_o_z_q * grasp_o_x_q;
        Vector3 gripper2_new_position = SetPreGraspPosition(gripper2_new_q, object_in_unity_position, dz);
        gripper2.transform.position = gripper2_new_position;
        gripper2.transform.rotation = gripper2_new_q;
        Goal_pose_update("custom"); // goal pose
    }

    public void Slider_distance_control(float slidervalue)
    {
        dz = -slidervalue*4 - 10f;
        Vector3 gripper2_new_position = SetPreGraspPosition(gripper2.transform.rotation, object_in_unity_position, dz);
        gripper2.transform.position = gripper2_new_position;
        Goal_pose_update("custom"); // goal pose
    }
    //===================================================slider functions=====================================

    // +++++++++++++++++++++++++++++++++++++++++++++++++Main script UI control++++++++++++++++++++++++++++++++++++++++++++++++++
    public void Save_inputfield()
    {
        IP = Input.text;
        Debug.Log("input saved" + IP);
    }

    public void Goal_pose_update(string gripper_name)
    {
        
        if (gripper_name =="Gripper1")
        {
            Debug.Log("set up goal as gripper 1 pose");
            Quaternion gripper_O_Y_q = Quaternion.Euler(new Vector3(0f, obj_DB.object_data_base[object_of_interest][0].x, 0f));
            Quaternion gripper_O_Z_q = Quaternion.Euler(new Vector3(0f, 0f, 90-obj_DB.object_data_base[object_of_interest][0].z));
            goal_pose.rotation = object_in_camera.rotation * gripper_O_Z_q * gripper_O_Y_q;
            goal_pose.position = SetPreGraspPosition(goal_pose.rotation, object_in_camera.position, -0.1f);
        }
        else if (gripper_name == "Gripper2")
        {
            Debug.Log("set up goal as gripper 2 pose");
            Quaternion gripper_O_Y_q = Quaternion.Euler(new Vector3(0f, obj_DB.object_data_base[object_of_interest][1].x, 0f));
            Quaternion gripper_O_Z_q = Quaternion.Euler(new Vector3(0f, 0f, 90-obj_DB.object_data_base[object_of_interest][1].z));
            goal_pose.rotation = object_in_camera.rotation * gripper_O_Z_q * gripper_O_Y_q;           
            goal_pose.position = SetPreGraspPosition(goal_pose.rotation, object_in_camera.position, -0.1f);
        }
        else // if (gripper_name == "custom")
        {
            Quaternion gripper_O_Y_q = Quaternion.Euler(new Vector3(0f, x_angle, 0f));
            Quaternion gripper_O_Z_q = Quaternion.Euler(new Vector3(0f, 0f, 90-z_angle));
            goal_pose.rotation = object_in_camera.rotation * gripper_O_Z_q * gripper_O_Y_q;
            goal_pose.position = SetPreGraspPosition(goal_pose.rotation, object_in_camera.position, dz * 0.01f);
        }
    }

    public void FeedbackUI(Vector2 feedback)
    {
        /*
            (1,0,count) found solution ask for permission for execution
            (0,0,count) no solution found ask for instructions 
            (1,1,count) execution finished reset UI
            (1,2,count) execution failed
            other hide
         */       
        if (feedback == new Vector2(1,0))
        {
            feedback_menu.transform.Find("btnConfirm").gameObject.SetActive(true);
            feedback_menu.transform.Find("btnCancel2").gameObject.SetActive(true);
            msgfeedback.text = "Solution found, execute?";
        }
        else if (feedback == new Vector2(0, 0))
        {
            feedback_menu.transform.Find("btnConfirm").gameObject.SetActive(false);
            feedback_menu.transform.Find("btnCancel2").gameObject.SetActive(true);
            msgfeedback.text = "No Solution found!";          
        }
        else if (feedback == new Vector2(1, 1))
        {
            feedback_menu.transform.Find("btnConfirm").gameObject.SetActive(false);
            feedback_menu.transform.Find("btnCancel2").gameObject.SetActive(true);
            msgfeedback.text = "Done";
        }
        else
        {
            feedback_menu.transform.Find("btnConfirm").gameObject.SetActive(false);
            feedback_menu.transform.Find("btnCancel2").gameObject.SetActive(true);
            msgfeedback.text = "Execution Failed";
        }
    }
    //public void GameObject_activation_switch(string name, string option="deactivate")
    //{
    //    GameObject gameObject = GameObject.Find(name);
    //    if (option == "activate")
    //    {
    //        gameObject.SetActive(true);
    //    }
    //    else
    //    {
    //        gameObject.SetActive(false);
    //    }
    //}

    public void Toggle_user_menu(Vector2 position, Vector2 normalized_position)
    {
        for (int i = 0; i < mainscript.obj_screen_location.Length; i++)
        {
            Vector3 obj_norm_p = Camera.main.ScreenToViewportPoint(mainscript.obj_screen_location[i]);
            if (Mathf.Abs(normalized_position.x - obj_norm_p.x) < 0.05f && Mathf.Abs(normalized_position.y - obj_norm_p.y) < 0.05f)
            {
                object_of_interest = mainscript.object_ids[i];
                object_in_array = i;
                object_in_camera = mainscript.pose_array[i];
                goal_pose.position = mainscript.pose_array[i].position;
                object_in_unity_position = Camera.main.ScreenToWorldPoint(new Vector3(mainscript.obj_screen_location[i].x, mainscript.obj_screen_location[i].y, 30f));
                btnGrasp.GetComponentInChildren<UnityEngine.UI.Text>().text = "Grasp";
            }
        }
        if (is_user_menu_open)
        {
            is_user_menu_open = !is_user_menu_open;
            user_menu.gameObject.SetActive(false);
            btnGrasp.GetComponentInChildren<UnityEngine.UI.Text>().text = "Show";
        }
        else
        {
            is_user_menu_open = !is_user_menu_open;
            user_menu.gameObject.transform.position = Camera.main.ScreenToWorldPoint(new Vector3(position.x, position.y, 17f));
            user_menu.gameObject.SetActive(true);
        }
    }

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // helper functions---------------
    private Quaternion SetGripperOrientation(Vector3 g_o_euler) // careful when use this since grasp_o_z_q will be reset
    {
        Quaternion gripper_o_z_q = Quaternion.Euler(new Vector3(0f, 0f, g_o_euler.z));
        //Debug.Log("rotation about z axis" + gripper_o_z_q);
        //Quaternion gripper_o_y_q = Quaternion.Euler(new Vector3(0f, g_o_euler.y, 0f));
        Quaternion gripper_o_x_q = Quaternion.Euler(new Vector3(g_o_euler.x, 0f, 0f));
        //Debug.Log("compare///////////////////////////////\n" + gripper_o_y_q * gripper_o_x_q * gripper_o_z_q);
        //Debug.Log(Quaternion.Euler(g_o_euler));
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< result show that quaternion.euler return euler YXZ rotation >>>>>>>>>>>>>>>>>>>>>
        //Quaternion test = O_K_Q;
        //Quaternion gripper_o_x_q = Quaternion.Euler(g_o_euler);
        Quaternion object_in_unity = InverseQuaternion(object_in_camera.rotation);
        Quaternion gripper_orientation = object_in_unity * gripper_o_z_q * gripper_o_x_q; //order matters zxy?zyx?
        //Debug.Log("quaternion * quaternion: " + test * O_K_Q);
        return gripper_orientation;
    }

    private Vector3 SetPreGraspPosition(Quaternion gripper_rotation, Vector3 gripper_center, float distance_z=-10f)
    {
        Vector3 pre_grasp_position;
        Matrix4x4 Gripper_R = Matrix4x4.Rotate(gripper_rotation);
        pre_grasp_position.x = gripper_center.x + Gripper_R.m02 * distance_z;
        pre_grasp_position.y = gripper_center.y + Gripper_R.m12 * distance_z;
        pre_grasp_position.z = gripper_center.z + Gripper_R.m22 * distance_z;
        return pre_grasp_position;
    }

    private void Initialize_grasp_tracking()
    {
        dz = -10f;
        x_angle = 180f;
        z_angle = -45f;
    }

    private Quaternion InverseQuaternion(Quaternion original_quaternion)
    {
        Quaternion inverse_quaternion = original_quaternion;
        inverse_quaternion.x = -inverse_quaternion.x;
        inverse_quaternion.y = -inverse_quaternion.y;
        inverse_quaternion.z = -inverse_quaternion.z;
        return inverse_quaternion;
    }

    public void ShowDefaultApply()
    {
        default_grasp_UI.transform.Find("btnApply2").gameObject.SetActive(true);
    }

    public void HighLightGripper(string name)
    {
        Color new_color = new Color(0.8f, 0.9f, 0.1f, 0.7f);
        Color original_color = new Color(0.4f, 0.85f, 0.6f, 0.5f);
        if (name == "Gripper1")
        {
            foreach (Renderer child_renderer in gripper1.GetComponentsInChildren<Renderer>())
            {
                child_renderer.material.color = new_color;
            }
        }
        else if(name == "Gripper2")
        {
            foreach (Renderer child_renderer in gripper2.GetComponentsInChildren<Renderer>())
            {
                child_renderer.material.color = new_color;
            }
        }
        else
        {
            foreach (Renderer child_renderer in gripper1.GetComponentsInChildren<Renderer>())
            {
                child_renderer.material.color = original_color;
            }

            foreach (Renderer child_renderer in gripper2.GetComponentsInChildren<Renderer>())
            {
                child_renderer.material.color = original_color;
            }
        }
        Goal_pose_update(name);
    }
}
