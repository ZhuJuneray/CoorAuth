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
        // ����һ�����ӻ���������һ���յ�GameObject�����ڱ�ʾĿ��ע�ӵ�
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

        if (Physics.Raycast(eyePosition, eyeDirection, out hit, Mathf.Infinity)) // �����ཻ
        {
            Vector3 gazePoint = hit.point; // ע�ӵ��λ��
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
