using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class GazeVisualization : MonoBehaviour
{
    [SerializeField]
    public GameObject leftObj, rightObj;
    public bool setGazeVisible;

    private GameObject male;
    private Transform leftTrans, rightTrans;
    private GameObject gazeVisualizationObject;
    private Transform tempTransform;
    private RaycastHit hit;
    void Start()
    {
        // 创建一个可视化对象，例如一个空的GameObject，用于表示目光注视点
        gazeVisualizationObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        gazeVisualizationObject.GetComponent<Renderer>().material.color = Color.red;
        gazeVisualizationObject.transform.localScale = new Vector3(0.02f, 0.02f, 0.02f);
        gazeVisualizationObject.SetActive(false);

        male = GameObject.Find("male7.17");
        tempTransform = new GameObject().transform;
    }

    void Update()
    {
        leftTrans = leftObj.transform;
        rightTrans = rightObj.transform;

        Quaternion headRot, leftEyeRot, rightEyeRot;
        Vector3 headPos, leftEyePos, rightEyePos;
        // 真实世界头 眼（）的参数
        OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.Head,
                NodeStatePropertyType.Position, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out headPos);
        OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.Head,
                NodeStatePropertyType.Orientation, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out headRot);

        OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.LeftEye,
                NodeStatePropertyType.Position, OVRPlugin.Node.EyeLeft, OVRPlugin.Step.Render, out leftEyePos);
        OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.LeftEye,
                NodeStatePropertyType.Orientation, OVRPlugin.Node.EyeLeft, OVRPlugin.Step.Render, out leftEyeRot);

        OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.RightEye,
                NodeStatePropertyType.Position, OVRPlugin.Node.EyeRight, OVRPlugin.Step.Render, out rightEyePos);
        OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.RightEye,
                NodeStatePropertyType.Orientation, OVRPlugin.Node.EyeRight, OVRPlugin.Step.Render, out rightEyeRot);
        //Debug.Log("phy pos" + headPos.ToString() + "headrot" + headRot.eulerAngles + "lefteyerot" + leftTrans.localRotation.eulerAngles + "righteyerot" + rightTrans.localRotation.eulerAngles);
        Vector3 rot3 = headRot.eulerAngles;
        //rot3.y -= 1;
        headRot = Quaternion.Euler(rot3);

        //if (Mathf.Abs(headRot.eulerAngles.y) < 7 && Mathf.Abs(headRot.eulerAngles.x) < 7
        //    && Mathf.Abs(leftTrans.localRotation.y) < 7 && Mathf.Abs(leftTrans.localRotation.y) < 7)
        //{
        //    Vector3 p = leftEyePos;
        //    p.y = 0;
        //    rot3.x -= 18;
        //    tempTransform.position = p;
        //    tempTransform.rotation = Quaternion.Euler(rot3);
        //}

        if (leftTrans.localRotation.y < 0)
        {
            tempTransform.position = leftEyePos;
            tempTransform.rotation = leftTrans.localRotation * Quaternion.Euler(rot3);
            //Debug.Log("yes");
        }
        else
        {
            tempTransform.position = rightEyePos;
            tempTransform.rotation = rightTrans.localRotation * Quaternion.Euler(rot3);
        }
        // 获取正前方射线方向
        Vector3 forwardDirection = tempTransform.forward;


        if (Physics.Raycast(tempTransform.position, forwardDirection, out hit, Mathf.Infinity)) // 射线相交
        {
            Vector3 gazePoint = hit.point; // 注视点的位置
            gazeVisualizationObject.transform.position = gazePoint;
            GameObject g = hit.collider.gameObject;
            if (!setGazeVisible) { return; }
            if (g.Equals(gazeVisualizationObject)) gazeVisualizationObject.SetActive(false);
            else gazeVisualizationObject.SetActive(true);
        }
        else
        {
            gazeVisualizationObject.SetActive(false);
        }
    }

    public Vector3 GetGazePos()
    {
        return hit.point;
    }

}
