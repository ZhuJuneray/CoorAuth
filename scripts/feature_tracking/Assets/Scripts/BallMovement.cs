using System.Collections;
using UnityEngine;

public class BallMovement : MonoBehaviour
{
    private Transform[] targetPoints; // �洢����Ŀ����Transform
    [SerializeField]
    private float totalTimeForMove = 1.0f; // ����ÿ���ƶ�����ʱ��
    [SerializeField]
    private float totalTimeForWait = 1.0f;
    private int currentTargetIndex = 0; // ��ǰĿ�������
    private float minScale = 0.05f;
    private float maxScale = 1f;

    private bool begin, init;
    private AuthController trackingScript;
    private GameObject[] spheres;  // ���ڴ洢���������
    void Start()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//��ȡcontroller
        this.begin = trackingScript.Begin;
        this.init = true;
        // ��ȡ��������������
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
        if (begin && init) // �״ε���
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
                end = GetNextTargetPosition(); // ��ÿһ֡������ end ��λ��
                // ʹ��Vector3.Lerpƽ���ƶ�
                transform.position = Vector3.Lerp(start, end, elapsedTime / totalTimeForMove);

                // ʹ��Mathf.Lerp���ȱ��/��С
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
                end = GetNextTargetPosition(); // ��ÿһ֡������ end ��λ��
                transform.position = end;

                float scale = Mathf.Lerp(maxScale, minScale, waitTimeElapsed);
                transform.localScale = new Vector3(scale, scale, scale);

                waitTimeElapsed += Time.deltaTime / totalTimeForMove;
                yield return null;
            }

            // �л�����һ��Ŀ���
            currentTargetIndex = (currentTargetIndex + 1) % targetPoints.Length;
            if (currentTargetIndex + 1 == targetPoints.Length)
                Debug.LogError("Wait---");
        }
    }

    Vector3 GetCurrentTargetPosition()
    {
        // �������ȡ��ǰʱ�̵�Ŀ���λ��
        if (currentTargetIndex < targetPoints.Length)
        {
            return targetPoints[currentTargetIndex].position;
        }
        else
        {
            // ��������������鳤�ȣ�����ԭ����ΪĬ��ֵ
            return Vector3.zero;
        }
    }

    Vector3 GetNextTargetPosition()
    {
        // �������ȡ��ǰʱ�̶�Ӧ����һ��Ŀ���λ��
        int nextTargetIndex = (currentTargetIndex + 1) % targetPoints.Length;
        if (nextTargetIndex < targetPoints.Length)
        {
            return targetPoints[nextTargetIndex].position;
        }
        else
        {
            // ��������������鳤�ȣ�����ԭ����ΪĬ��ֵ
            return Vector3.zero;
        }
    }
}
