using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class GazeVisualization : MonoBehaviour
{
    [SerializeField]
    public GameObject eyeObject;

    private Transform gazeTransform;
    private GameObject gazeVisualizationObject;

    void Start()
    {
        // 创建一个可视化对象，例如一个空的GameObject，用于表示目光注视点
        gazeVisualizationObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        gazeVisualizationObject.GetComponent<Renderer>().material.color = Color.red;
        gazeVisualizationObject.transform.localScale = new Vector3(0.02f, 0.02f, 0.02f);
        gazeVisualizationObject.SetActive(false);

    }

    void Update()
    {
        gazeTransform = eyeObject.transform;
        Vector3 eyePosition = gazeTransform.position;
        Vector3 eyeDirection = gazeTransform.forward;
        RaycastHit hit;

        if (Physics.Raycast(eyePosition, eyeDirection, out hit, Mathf.Infinity)) // 射线相交
        {
            Vector3 gazePoint = hit.point; // 注视点的位置
            gazeVisualizationObject.transform.position = gazePoint;
            GameObject g = hit.collider.gameObject;
            if (g.Equals(gazeVisualizationObject)) gazeVisualizationObject.SetActive(false);
            else gazeVisualizationObject.SetActive(true);
        }
        else
        {
            gazeVisualizationObject.SetActive(false);
        }
    }
}
