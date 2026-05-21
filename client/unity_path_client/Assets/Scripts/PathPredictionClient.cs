using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class PathPredictionClient : MonoBehaviour
{
    [Header("Server")]
    [SerializeField] private string serverUrl = "http://127.0.0.1:8000/predict";
    [SerializeField] private float requestIntervalSeconds = 0.25f;

    [Header("Scene")]
    [SerializeField] private Transform agentRoot;
    [SerializeField] private Camera sourceCamera;
    [SerializeField] private PathPredictionVisualizer visualizer;

    [Header("History")]
    [SerializeField] private int historyLength = 5;
    [SerializeField] private bool sendRgbFrames = false;
    [SerializeField] private int captureSize = 128;
    [SerializeField] private bool demoMode = true;
    [SerializeField] private bool fallbackToLocalMotion = true;
    [SerializeField] private float stationaryDemoStep = 0.75f;

    [Header("Rendering")]
    [SerializeField] private float pathHeight = 0.05f;
    [SerializeField] private float pathScale = 1.0f;

    private readonly Queue<EgoMotionSample> egoHistory = new Queue<EgoMotionSample>();
    private readonly Queue<string> rgbHistory = new Queue<string>();
    private Vector3 previousPosition;
    private Quaternion previousRotation;
    private bool hasPreviousPose;
    private bool requestInFlight;
    private float nextRequestTime;

    public void ConfigureRuntime(
        Transform root,
        Camera camera,
        PathPredictionVisualizer pathVisualizer,
        string predictionServerUrl,
        bool useDemoMode
    )
    {
        agentRoot = root != null ? root : transform;
        sourceCamera = camera;
        visualizer = pathVisualizer != null ? pathVisualizer : GetComponent<PathPredictionVisualizer>();
        serverUrl = string.IsNullOrWhiteSpace(predictionServerUrl) ? serverUrl : predictionServerUrl;
        demoMode = useDemoMode;
        if (visualizer == null)
        {
            visualizer = gameObject.AddComponent<PathPredictionVisualizer>();
        }
        visualizer.Configure(agentRoot, pathScale, pathHeight);
    }

    private void Reset()
    {
        agentRoot = transform;
        sourceCamera = Camera.main;
        visualizer = GetComponent<PathPredictionVisualizer>();
    }

    private void Awake()
    {
        if (agentRoot == null)
        {
            agentRoot = transform;
        }
        if (visualizer == null)
        {
            visualizer = GetComponent<PathPredictionVisualizer>();
        }
        if (visualizer == null)
        {
            visualizer = gameObject.AddComponent<PathPredictionVisualizer>();
        }
        visualizer.Configure(agentRoot, pathScale, pathHeight);
    }

    private void Update()
    {
        RecordEgoMotion();
        if (sendRgbFrames && sourceCamera != null)
        {
            EnqueueBounded(rgbHistory, CaptureCameraPngBase64(), historyLength);
        }

        if (!requestInFlight && Time.time >= nextRequestTime && egoHistory.Count >= historyLength)
        {
            nextRequestTime = Time.time + requestIntervalSeconds;
            if (demoMode)
            {
                RenderPath(BuildLocalMotionPrediction());
            }
            else
            {
                StartCoroutine(RequestPrediction());
            }
        }
    }

    private void RecordEgoMotion()
    {
        if (!hasPreviousPose)
        {
            previousPosition = agentRoot.position;
            previousRotation = agentRoot.rotation;
            hasPreviousPose = true;
            return;
        }

        Vector3 worldDelta = agentRoot.position - previousPosition;
        Vector3 localDelta = Quaternion.Inverse(previousRotation) * worldDelta;
        Quaternion deltaRotation = Quaternion.Inverse(previousRotation) * agentRoot.rotation;
        float yawRadians = Mathf.DeltaAngle(0.0f, deltaRotation.eulerAngles.y) * Mathf.Deg2Rad;

        EnqueueBounded(
            egoHistory,
            new EgoMotionSample(localDelta.z, localDelta.x, yawRadians),
            historyLength
        );

        previousPosition = agentRoot.position;
        previousRotation = agentRoot.rotation;
    }

    private IEnumerator RequestPrediction()
    {
        requestInFlight = true;
        string payload = BuildRequestJson();
        byte[] body = Encoding.UTF8.GetBytes(payload);

        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(body);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                PathPredictionResponse response = JsonUtility.FromJson<PathPredictionResponse>(
                    request.downloadHandler.text
                );
                RenderPath(response);
            }
            else
            {
                Debug.LogWarning($"Path prediction request failed: {request.error}");
                if (fallbackToLocalMotion)
                {
                    RenderPath(BuildLocalMotionPrediction());
                }
            }
        }

        requestInFlight = false;
    }

    private string BuildRequestJson()
    {
        StringBuilder builder = new StringBuilder(4096);
        builder.Append("{\"ego_history\":[");
        int index = 0;
        foreach (EgoMotionSample sample in egoHistory)
        {
            if (index++ > 0)
            {
                builder.Append(',');
            }
            builder.Append('[');
            AppendFloat(builder, sample.forward);
            builder.Append(',');
            AppendFloat(builder, sample.right);
            builder.Append(',');
            AppendFloat(builder, sample.yaw);
            builder.Append(']');
        }
        builder.Append(']');

        if (sendRgbFrames && rgbHistory.Count == historyLength)
        {
            builder.Append(",\"rgb_frames\":[");
            index = 0;
            foreach (string encodedFrame in rgbHistory)
            {
                if (index++ > 0)
                {
                    builder.Append(',');
                }
                builder.Append('"');
                builder.Append(encodedFrame);
                builder.Append('"');
            }
            builder.Append(']');
            builder.Append(",\"image_size\":");
            builder.Append(captureSize);
        }

        builder.Append('}');
        return builder.ToString();
    }

    private void RenderPath(PathPredictionResponse response)
    {
        if (visualizer == null)
        {
            return;
        }

        visualizer.Configure(agentRoot, pathScale, pathHeight);
        visualizer.Render(response);
    }

    private PathPredictionResponse BuildLocalMotionPrediction()
    {
        Vector2 velocity = AverageRecentLocalVelocity();
        if (velocity.sqrMagnitude < 0.0001f)
        {
            velocity = new Vector2(stationaryDemoStep, 0.0f);
        }
        PathPoint[] path = new PathPoint[historyLength];
        for (int i = 0; i < path.Length; i++)
        {
            float step = i + 1;
            path[i] = new PathPoint
            {
                forward = velocity.x * step,
                right = velocity.y * step
            };
        }

        return new PathPredictionResponse
        {
            future_steps = path.Length,
            selected_mode = 0,
            path = path,
            mode_confidences = new[] { 1.0f }
        };
    }

    private Vector2 AverageRecentLocalVelocity()
    {
        if (egoHistory.Count == 0)
        {
            return Vector2.zero;
        }

        float forward = 0.0f;
        float right = 0.0f;
        foreach (EgoMotionSample sample in egoHistory)
        {
            forward += sample.forward;
            right += sample.right;
        }
        float count = egoHistory.Count;
        return new Vector2(forward / count, right / count);
    }

    private string CaptureCameraPngBase64()
    {
        RenderTexture previousTarget = sourceCamera.targetTexture;
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture target = RenderTexture.GetTemporary(captureSize, captureSize, 24, RenderTextureFormat.ARGB32);
        Texture2D texture = new Texture2D(captureSize, captureSize, TextureFormat.RGB24, false);

        try
        {
            sourceCamera.targetTexture = target;
            RenderTexture.active = target;
            sourceCamera.Render();
            texture.ReadPixels(new Rect(0, 0, captureSize, captureSize), 0, 0);
            texture.Apply();
            return Convert.ToBase64String(texture.EncodeToPNG());
        }
        finally
        {
            sourceCamera.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            RenderTexture.ReleaseTemporary(target);
            Destroy(texture);
        }
    }

    private static void EnqueueBounded<T>(Queue<T> queue, T value, int maxCount)
    {
        queue.Enqueue(value);
        while (queue.Count > maxCount)
        {
            queue.Dequeue();
        }
    }

    private static void AppendFloat(StringBuilder builder, float value)
    {
        builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }
}
