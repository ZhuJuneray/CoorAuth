using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class GazeFeaturesTracking : MonoBehaviour
{
    [SerializeField]
    [Range(0f, 1f)]
    public float threshold = 0.2f;
    private OVRPlugin.EyeGazesState _currentEyeGazesState;
    private GameObject leftEye;
    private GameObject rightEye;
    private AuthController trackingScript;
    private string Name;
    private bool Begin;
    private string pathStr;
    private List<GazeFeatures> gazeFeaturesCalculate, gazeFeaturesRaw;
    private void Awake()
    {

    }
    private void Start()
    {
        leftEye = GameObject.Find("LeftEye");
        rightEye = GameObject.Find("RightEye");
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();
        if (trackingScript)
        {
            Name = trackingScript.Name;
            Begin = trackingScript.Begin;
        }
        gazeFeaturesCalculate = new();
        gazeFeaturesRaw = new();
    }
    private void Update()
    {
        Begin = trackingScript.Begin;
        Name = trackingScript.Name + "-" + trackingScript.indexOfSize.ToString() + "-" + trackingScript.indexOfPIN.ToString()
            + "-" + trackingScript.time.ToString();
        if (!Begin && gazeFeaturesCalculate.Count > 5) // 有数据
        {
            Debug.Log("name" + Name);
            FileInput();
            // 清空
            gazeFeaturesCalculate.Clear();
            gazeFeaturesRaw.Clear();
        }

    }
    private void FixedUpdate()
    {
        // 更新
        if (!OVRPlugin.GetEyeGazesState(OVRPlugin.Step.Render, -1, ref _currentEyeGazesState))
            return;
        var eyeGazeLeft = _currentEyeGazesState.EyeGazes[0];
        var eyeGazeRight = _currentEyeGazesState.EyeGazes[1];
        if (!eyeGazeLeft.IsValid || !eyeGazeRight.IsValid)
            return;
        if (eyeGazeLeft.Confidence < threshold || eyeGazeRight.Confidence < threshold)
            return;
        var poseLeft = eyeGazeLeft.Pose.ToOVRPose();
        var poseRight = eyeGazeRight.Pose.ToOVRPose();
        //转换到头部坐标架
        poseLeft = poseLeft.ToHeadSpacePose();
        poseRight = poseRight.ToHeadSpacePose();

        if (Begin)
        {
            GazeFeatures g = new GazeFeatures(leftEye.transform.rotation, rightEye.transform.rotation);
            gazeFeaturesCalculate.Add(g);
            g = new GazeFeatures(poseLeft.orientation, poseRight.orientation);
            gazeFeaturesRaw.Add(g);
        }
        
    }

    private void OnDestroy()
    {
        //FileInput();
    }

    public void FileInput()
    {
        pathStr = @"E:\Desktop\data\VRAuth1\GazeCalculate_data_" + Name + ".csv";
        using (StreamWriter writer = new StreamWriter(pathStr))
        {
            // 写入CSV文件的标题行
            writer.WriteLine("L-QuaternionX,L-QuaternionY,L-QuaternionZ,L-QuaternionW,R-QuaternionX,R-QuaternionY,R-QuaternionZ,R-QuaternionW,");
            // 写入数据行
            foreach (var data in gazeFeaturesCalculate)
            {
                writer.WriteLine($"{data.LeftQuat.x},{data.LeftQuat.y},{data.LeftQuat.z},{data.LeftQuat.w},{data.RightQuat.x},{data.RightQuat.y},{data.RightQuat.z},{data.RightQuat.w}");
            }
            Debug.Log("Finish Writing GazeFeaturesCal");
        }

        pathStr = @"E:\Desktop\data\VRAuth1\GazeRaw_data_" + Name + ".csv";
        using (StreamWriter writer = new StreamWriter(pathStr))
        {
            // 写入CSV文件的标题行
            writer.WriteLine("L-QuaternionX,L-QuaternionY,L-QuaternionZ,L-QuaternionW,R-QuaternionX,R-QuaternionY,R-QuaternionZ,R-QuaternionW,");
            // 写入数据行
            foreach (var data in gazeFeaturesRaw)
            {
                writer.WriteLine($"{data.LeftQuat.x},{data.LeftQuat.y},{data.LeftQuat.z},{data.LeftQuat.w},{data.RightQuat.x},{data.RightQuat.y},{data.RightQuat.z},{data.RightQuat.w}");
            }
            Debug.Log("Finish Writing GazeFeaturesRaw");
        }
    }
}

public class GazeFeatures
{
    public Quaternion LeftQuat { get; set; }
    public Quaternion RightQuat { get; set; }
    public GazeFeatures(Quaternion l, Quaternion r)
    {
        LeftQuat = l;
        RightQuat = r;
    }
}
