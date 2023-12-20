using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using OVR;

public class HeadFeaturesTracking : MonoBehaviour
{
    public OVRSkeleton skeleton;
    private Transform transHead, transNeck;
    private string pathHead, pathNeck;
    private AuthController trackingScript;
    private string Name;
    private bool Begin;
    private List<InterestedFeature> headFeatures;
    private List<InterestedFeature> neckFeatures;

    // Start is called before the first frame update
    void Start()
    {
        if (skeleton == null)
        {
            Debug.LogError("Empty skeleton");
        }
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//获取controller
        if (trackingScript)
        {
            Name = trackingScript.Name;
            Begin = trackingScript.Begin;
        }
        headFeatures = new();
        neckFeatures = new();
    }

    // Update is called once per frame
    void Update()
    {
        foreach (var b in skeleton.Bones)
        {
            if (b.Id == OVRSkeleton.BoneId.Body_Head)
            {
                transHead = b.Transform;
                // Debug.Log("transform of head" + trans.position);
            }
            else if (b.Id == OVRSkeleton.BoneId.Body_Neck)
            {
                transNeck = b.Transform;
            }
        }

        Begin = trackingScript.Begin;
        Name = trackingScript.Name + "-" + trackingScript.indexOfSize.ToString() + "-" + trackingScript.indexOfPIN.ToString()
            + "-" + trackingScript.time.ToString();

        if (!Begin && headFeatures.Count > 5) // 有数据
        {
            Debug.Log("name" + Name);
            FileInput();
            // 清空
            headFeatures.Clear();
            neckFeatures.Clear();
        }
    }

    private void FixedUpdate()
    {
        if (Begin)
        {
            InterestedFeature h = new InterestedFeature(transHead.position, transHead.rotation);
            headFeatures.Add(h);
            //Quaternion rot;
            //Vector3 pos;
            //// 真实世界头的参数
            //OVRNodeStateProperties.GetNodeStatePropertyVector3(UnityEngine.XR.XRNode.Head,
            //        NodeStatePropertyType.Position, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out pos);
            //OVRNodeStateProperties.GetNodeStatePropertyQuaternion(UnityEngine.XR.XRNode.Head,
            //        NodeStatePropertyType.Orientation, OVRPlugin.Node.Head, OVRPlugin.Step.Render, out rot);
            //Debug.Log("trans pos" + transHead.position.ToString() + "rot" + transHead.rotation.eulerAngles + "phy pos" + pos.ToString()
            //    + "rot" + rot.eulerAngles);
            h = new InterestedFeature(transNeck.position, transNeck.rotation);
            neckFeatures.Add(h);
        }
    }
    private void OnDestroy()
    {
        //FileInput();

    }

    public void FileInput()
    {
        pathHead = @"E:\Desktop\data\VRAuth1\Head_data_" + Name + ".csv";
        pathNeck = @"E:\Desktop\data\VRAuth1\Neck_data_" + Name + ".csv";
        using (StreamWriter writer = new StreamWriter(pathHead))
        {
            // 写入CSV文件的标题行
            writer.WriteLine("H-Vector3X,H-Vector3Y,H-Vector3Z,H-QuaternionX,H-QuaternionY,H-QuaternionZ,H-QuaternionW");
            // 写入数据行
            foreach (var data in headFeatures)
            {
                writer.WriteLine($"{data.Vector3Data.x},{data.Vector3Data.y},{data.Vector3Data.z},{data.QuaternionData.x},{data.QuaternionData.y},{data.QuaternionData.z},{data.QuaternionData.w}");
            }

            Debug.Log("Finish Writing HeadFeatures");
        }

        using (StreamWriter writer = new StreamWriter(pathNeck))
        {
            // 写入CSV文件的标题行
            writer.WriteLine("N-Vector3X,N-Vector3Y,N-Vector3Z,N-QuaternionX,N-QuaternionY,N-QuaternionZ,N-QuaternionW");
            // 写入数据行
            foreach (var data in neckFeatures)
            {
                writer.WriteLine($"{data.Vector3Data.x},{data.Vector3Data.y},{data.Vector3Data.z},{data.QuaternionData.x},{data.QuaternionData.y},{data.QuaternionData.z},{data.QuaternionData.w}");
            }

            Debug.Log("Finish Writing NeckFeatures");
        }
    }

}

public class InterestedFeature
{
    public Vector3 Vector3Data { get; set; }
    public Quaternion QuaternionData { get; set; }
    public InterestedFeature(Vector3 v, Quaternion q)
    {
        Vector3Data = v;
        QuaternionData = q;
    }
}