using System.Collections;
using UnityEngine;

public class BallMovement : MonoBehaviour
{
    private Transform[] targetPoints; // 存储六个目标点的Transform
    [SerializeField]
    private float totalTimeForMove = 1.0f; // 控制每次移动的总时间
    [SerializeField]
    private float totalTimeForWait = 1.0f;
    private int currentTargetIndex = 0; // 当前目标点索引
    private float minScale = 0.05f;
    private float maxScale = 1f;

    private bool begin, init;
    private AuthController trackingScript;
    private GameObject[] spheres;  // 用于存储球体的数组
    void Start()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//获取controller
        this.begin = trackingScript.Begin;
        this.init = true;
        // 获取所有子物体球体
        Transform t = GameObject.Find("GazeCalibration").transform;
        spheres = new GameObject[t.childCount];
        targetPoints = new Transform[t.childCount];
        for (int i = 0; i < t.childCount; i++)
        {
            spheres[i] = t.GetChild(i).gameObject;
            targetPoints[i] = spheres[i].transform;
        }
        
    }

    private void Update()
    {
        this.begin = trackingScript.Begin;
        if (begin && init) // 首次调用
        {
            transform.localScale = new Vector3(minScale, minScale, minScale);
            StartCoroutine(MoveToTargets());
            init = false;
        }
    }

    

    IEnumerator MoveToTargets()
    {
        while (true)
        {
            Vector3 start = transform.position;
            Vector3 end = targetPoints[currentTargetIndex].position;

            float elapsedTime = 0f;

            while (elapsedTime < totalTimeForMove)
            {
                //Debug.Log("Move" + currentTargetIndex);
                end = GetNextTargetPosition(); // 在每一帧都更新 end 的位置
                // 使用Vector3.Lerp平滑移动
                transform.position = Vector3.Lerp(start, end, elapsedTime / totalTimeForMove);

                // 使用Mathf.Lerp均匀变大/缩小
                float scale = Mathf.Lerp(minScale, maxScale, elapsedTime / totalTimeForMove);
                transform.localScale = new Vector3(scale, scale, scale);

                elapsedTime += Time.deltaTime;
                yield return null;
            }

            Debug.Log("currentIndex" + currentTargetIndex);
            float waitTimeElapsed = 0f;

            while (waitTimeElapsed < totalTimeForWait)
            {
                //Debug.Log("wait" + currentTargetIndex);
                end = GetNextTargetPosition(); // 在每一帧都更新 end 的位置
                transform.position = end;

                float scale = Mathf.Lerp(maxScale, minScale, waitTimeElapsed);
                transform.localScale = new Vector3(scale, scale, scale);

                waitTimeElapsed += Time.deltaTime / totalTimeForMove;
                yield return null;
            }

            // 切换到下一个目标点
            currentTargetIndex = (currentTargetIndex + 1) % targetPoints.Length;
            if (currentTargetIndex + 1 == targetPoints.Length)
                Debug.LogError("Wait---");
        }
    }

    Vector3 GetCurrentTargetPosition()
    {
        // 在这里获取当前时刻的目标点位置
        if (currentTargetIndex < targetPoints.Length)
        {
            return targetPoints[currentTargetIndex].position;
        }
        else
        {
            // 如果索引超出数组长度，返回原点作为默认值
            return Vector3.zero;
        }
    }

    Vector3 GetNextTargetPosition()
    {
        // 在这里获取当前时刻对应的下一个目标点位置
        int nextTargetIndex = (currentTargetIndex + 1) % targetPoints.Length;
        if (nextTargetIndex < targetPoints.Length)
        {
            return targetPoints[nextTargetIndex].position;
        }
        else
        {
            // 如果索引超出数组长度，返回原点作为默认值
            return Vector3.zero;
        }
    }
}
