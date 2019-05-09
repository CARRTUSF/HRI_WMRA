using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectDataBase : MonoBehaviour {
    // euler.x = rotation angle about X axis 
    // euler.y = rotation angle about y axis
    // euler.z = rotation angle about Z axis
    public IDictionary<string, List<Vector3>> object_data_base = new Dictionary<string, List<Vector3>>()
    {
        //can h=0.12m
        { "coke", new List<Vector3>()
            {
               new Vector3(90f, 0f, -45f),
               new Vector3(180f, 0f, -45f)
            }
        },
        //ball r=0.03m
        { "ball_3", new List<Vector3>()
            {
                new Vector3(135f, 0f, -45f),
                new Vector3(180f, 0f, -45f)
            }
        }
    };
}
