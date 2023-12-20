using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GazePositionCalculate : MonoBehaviour
{
    [SerializeField]
    public GameObject eyeObject;
    public OVRSkeleton skeleton;
    public float distance = 4.0f;

    private Transform gazeTransform, headTransform;
    private GameObject gazeCalibration;
    private AuthController trackingScript;
    private bool begin;

    // Start is called before the first frame update
    void Start()
    {
        gazeCalibration = GameObject.Find("GazeCalibration");
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//��ȡcontroller
        this.begin = trackingScript.Begin;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void FixedUpdate()
    {
        this.begin = trackingScript.Begin;
        gazeTransform = eyeObject.transform;
        Vector3 eyePosition = gazeTransform.position;
        Vector3 eyeDirection = gazeTransform.forward;
        Vector3 targetPoint = new();
        if (!begin)
        {
            return;
        }
        // Ѱ��male������ͷ������
        foreach (var b in skeleton.Bones)
        {
            if (b.Id == OVRSkeleton.BoneId.Body_Head)
            {
                headTransform = b.Transform;
            }
        }
        Quaternion rot;
        Vector3 pos;
        // ��ʵ����ͷ�Ĳ���
        OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.Head,
                NodeStatePropertyType.Position, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out pos);
        OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.Head,
                NodeStatePropertyType.Orientation, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out rot);

        //if (headTransform != null)
        //    Debug.Log("headpos" + headTransform.position.ToString() +
        //        "headforward" + headTransform.forward.ToString() + 
        //        "headrotation" + headTransform.rotation.eulerAngles.ToString() + 
        //        "eyeforward" + eyeDirection.ToString() +
        //        "calculatepos" + targetPoint.ToString() + "pos" + gazeCalibration.transform.position.ToString());
        // ����һ���յ� GameObject
        GameObject emptyObject = new GameObject("PhysicalHeadObject");

        // ��ȡ�� GameObject �� Transform ���
        Transform physicalHeadTransform = emptyObject.transform;
        // �޸�����(����x��ת��)�Ƕ�����
        Vector3 rotDuplicated = rot.eulerAngles;
        rotDuplicated.x += 25;
        rotDuplicated.y += 3;
        rot = Quaternion.Euler(rotDuplicated);
        physicalHeadTransform.position = pos;
        physicalHeadTransform.rotation = rot;
        //Debug.Log("pos" + pos.ToString() + "rot" + rot.eulerAngles.ToString() + "forward" + physicalHeadTransform.forward.ToString());
        targetPoint = pos + physicalHeadTransform.forward * distance;

        gazeCalibration.transform.position = targetPoint;
        gazeCalibration.transform.rotation = rot;
    }

}
