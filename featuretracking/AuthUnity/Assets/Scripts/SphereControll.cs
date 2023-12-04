using UnityEngine;

public class SphereControll : MonoBehaviour
{
    private GameObject[] spheres;  // ���ڴ洢�ĸ����������
    private int currentIndex = 0;  // ��ǰ��ʾ����������
    private bool begin, init;
    private AuthController trackingScript;

    void Start()
    {
        trackingScript = GameObject.Find("Controller").GetComponent<AuthController>();//��ȡcontroller
        this.begin = trackingScript.Begin;
        this.init = true;
        // ��ȡSphere�µ���������������
        spheres = new GameObject[transform.childCount];
        for (int i = 0; i < transform.childCount; i++)
        {
            spheres[i] = transform.GetChild(i).gameObject;
        }

    }

    private void Update()
    {
        this.begin = trackingScript.Begin;
        if (begin && init)
        {
            foreach (var s in spheres)
            {
                s.SetActive(false);
            }
            // ��ȡ��һ�������Renderer���
            Renderer currentRenderer = spheres[0].GetComponent<Renderer>();
            currentRenderer.material.color = Color.red;
            spheres[0].SetActive(true);
            // ������ʱ����ÿ��һ�����ShowNextSphere����
            InvokeRepeating("ShowNextSphere", 2f, 2f);
            init = false;
        }
    }

    void ShowNextSphere()
    {
        
        spheres[currentIndex].SetActive(false);
        currentIndex = (currentIndex + 1) % spheres.Length;
        spheres[currentIndex].SetActive(true);
        // ��ȡ��ǰ�����Renderer���
        Renderer currentRenderer = spheres[currentIndex].GetComponent<Renderer>();
        currentRenderer.material.color = Color.red;
    }
}
