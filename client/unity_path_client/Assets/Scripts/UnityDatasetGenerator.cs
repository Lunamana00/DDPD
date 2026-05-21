using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class UnityDatasetGenerator : MonoBehaviour
{
    [Header("Output")]
    [SerializeField] private bool generateOnStart;
    [SerializeField] private string runId = "unity_procedural_001";
    [SerializeField] private string outputRoot = "DDPDUnityDataset";
    [SerializeField] private bool usePersistentDataPath = true;
    [SerializeField] private bool overwriteExistingRun = true;

    [Header("Dataset")]
    [SerializeField] private int seed = 7001;
    [SerializeField] private int episodeCount = 8;
    [SerializeField] private int framesPerEpisode = 260;
    [SerializeField] private float sampleFps = 10.0f;
    [SerializeField] private int frameSkip = 1;
    [SerializeField] private int captureWidth = 128;
    [SerializeField] private int captureHeight = 128;
    [SerializeField] private float agentSpeedMetersPerSecond = 2.2f;

    [Header("Scene")]
    [SerializeField] private int routeWaypoints = 12;
    [SerializeField] private float waypointSpacingMin = 4.2f;
    [SerializeField] private float waypointSpacingMax = 7.0f;
    [SerializeField] private float laneWidth = 2.2f;
    [SerializeField] private float corridorHalfWidth = 5.4f;
    [SerializeField] private bool showDebugRouteLine;
    [SerializeField] private bool quitAfterGeneration;

    private readonly List<EpisodeSummary> summaries = new List<EpisodeSummary>();
    private GameObject generatedRoot;
    private GameObject agent;
    private Camera captureCamera;
    private Material[] palette;
    private bool isGenerating;

    private struct PoseRecord
    {
        public float x;
        public float y;
        public float angle;
    }

    private struct EpisodeSummary
    {
        public string episodeId;
        public int numSteps;
        public int seed;
        public float routeLength;
    }

    private void Start()
    {
        if (generateOnStart)
        {
            StartCoroutine(GenerateDataset());
        }
    }

    [ContextMenu("Generate WIT-VZ Raw Dataset")]
    public void GenerateFromContextMenu()
    {
        if (!isGenerating)
        {
            StartCoroutine(GenerateDataset());
        }
    }

    public void ConfigureOutput(
        string newRunId,
        string newOutputRoot,
        bool newUsePersistentDataPath,
        int newEpisodeCount,
        int newFramesPerEpisode,
        int newCaptureWidth,
        int newCaptureHeight
    )
    {
        runId = string.IsNullOrWhiteSpace(newRunId) ? runId : newRunId;
        outputRoot = string.IsNullOrWhiteSpace(newOutputRoot) ? outputRoot : newOutputRoot;
        usePersistentDataPath = newUsePersistentDataPath;
        episodeCount = Mathf.Max(1, newEpisodeCount);
        framesPerEpisode = Mathf.Max(8, newFramesPerEpisode);
        captureWidth = Mathf.Max(32, newCaptureWidth);
        captureHeight = Mathf.Max(32, newCaptureHeight);
    }

    public IEnumerator GenerateDataset()
    {
        if (isGenerating)
        {
            yield break;
        }
        isGenerating = true;
        summaries.Clear();

        string runDirectory = ResolveRunDirectory();
        if (Directory.Exists(runDirectory) && overwriteExistingRun)
        {
            Directory.Delete(runDirectory, true);
        }
        Directory.CreateDirectory(runDirectory);

        for (int episodeIndex = 1; episodeIndex <= episodeCount; episodeIndex++)
        {
            int episodeSeed = seed + episodeIndex * 1009;
            yield return GenerateEpisode(runDirectory, episodeIndex, episodeSeed);
        }

        WriteManifest(runDirectory);
        Debug.Log($"DDPD Unity dataset written to: {runDirectory}");
        isGenerating = false;

        if (quitAfterGeneration)
        {
            Application.Quit();
        }
    }

    public string GenerateDatasetBlocking()
    {
        if (isGenerating)
        {
            return ResolveRunDirectory();
        }
        isGenerating = true;
        summaries.Clear();

        string runDirectory = ResolveRunDirectory();
        if (Directory.Exists(runDirectory) && overwriteExistingRun)
        {
            Directory.Delete(runDirectory, true);
        }
        Directory.CreateDirectory(runDirectory);

        for (int episodeIndex = 1; episodeIndex <= episodeCount; episodeIndex++)
        {
            int episodeSeed = seed + episodeIndex * 1009;
            GenerateEpisodeBlocking(runDirectory, episodeIndex, episodeSeed);
        }

        WriteManifest(runDirectory);
        Debug.Log($"DDPD Unity dataset written to: {runDirectory}");
        isGenerating = false;
        return runDirectory;
    }

    private IEnumerator GenerateEpisode(string runDirectory, int episodeIndex, int episodeSeed)
    {
        string episodeId = $"episode_{episodeIndex:0000}";
        string episodeDirectory = Path.Combine(runDirectory, "episodes", episodeId);
        string rgbDirectory = Path.Combine(episodeDirectory, "rgb");
        Directory.CreateDirectory(rgbDirectory);

        System.Random rng = new System.Random(episodeSeed);
        List<Vector3> route = GenerateRoute(rng);
        float routeLength = ComputeRouteLength(route);
        BuildScene(route, rng);
        BuildAgent(route[0], route[1]);

        List<string> stepLines = new List<string>(framesPerEpisode);
        bool hasPreviousPose = false;
        PoseRecord previousPose = default;

        for (int step = 0; step < framesPerEpisode; step++)
        {
            float elapsed = step / Mathf.Max(sampleFps, 0.001f);
            float distance = Mathf.Min(elapsed * agentSpeedMetersPerSecond, Mathf.Max(0.0f, routeLength - 0.01f));
            PoseRoute(route, distance, out Vector3 position, out Quaternion rotation);
            agent.transform.SetPositionAndRotation(position, rotation);

            yield return new WaitForEndOfFrame();

            string frameRelativePath = $"episodes/{episodeId}/rgb/frame_{step:000000}.png";
            string frameAbsolutePath = Path.Combine(runDirectory, frameRelativePath.Replace('/', Path.DirectorySeparatorChar));
            CapturePng(frameAbsolutePath);

            PoseRecord currentPose = ToPoseRecord(agent.transform);
            string line = BuildStepJson(step, elapsed, frameRelativePath, currentPose, hasPreviousPose, previousPose, episodeSeed);
            stepLines.Add(line);
            previousPose = currentPose;
            hasPreviousPose = true;
        }

        File.WriteAllLines(Path.Combine(episodeDirectory, "steps.jsonl"), stepLines, Encoding.UTF8);
        File.WriteAllText(
            Path.Combine(episodeDirectory, "summary.json"),
            BuildEpisodeSummaryJson(episodeId, episodeSeed, framesPerEpisode, routeLength),
            Encoding.UTF8
        );

        summaries.Add(new EpisodeSummary
        {
            episodeId = episodeId,
            numSteps = framesPerEpisode,
            seed = episodeSeed,
            routeLength = routeLength
        });
    }

    private void GenerateEpisodeBlocking(string runDirectory, int episodeIndex, int episodeSeed)
    {
        string episodeId = $"episode_{episodeIndex:0000}";
        string episodeDirectory = Path.Combine(runDirectory, "episodes", episodeId);
        string rgbDirectory = Path.Combine(episodeDirectory, "rgb");
        Directory.CreateDirectory(rgbDirectory);

        System.Random rng = new System.Random(episodeSeed);
        List<Vector3> route = GenerateRoute(rng);
        float routeLength = ComputeRouteLength(route);
        BuildScene(route, rng);
        BuildAgent(route[0], route[1]);

        List<string> stepLines = new List<string>(framesPerEpisode);
        bool hasPreviousPose = false;
        PoseRecord previousPose = default;

        for (int step = 0; step < framesPerEpisode; step++)
        {
            float elapsed = step / Mathf.Max(sampleFps, 0.001f);
            float distance = Mathf.Min(elapsed * agentSpeedMetersPerSecond, Mathf.Max(0.0f, routeLength - 0.01f));
            PoseRoute(route, distance, out Vector3 position, out Quaternion rotation);
            agent.transform.SetPositionAndRotation(position, rotation);

            string frameRelativePath = $"episodes/{episodeId}/rgb/frame_{step:000000}.png";
            string frameAbsolutePath = Path.Combine(runDirectory, frameRelativePath.Replace('/', Path.DirectorySeparatorChar));
            CapturePng(frameAbsolutePath);

            PoseRecord currentPose = ToPoseRecord(agent.transform);
            string line = BuildStepJson(step, elapsed, frameRelativePath, currentPose, hasPreviousPose, previousPose, episodeSeed);
            stepLines.Add(line);
            previousPose = currentPose;
            hasPreviousPose = true;
        }

        File.WriteAllLines(Path.Combine(episodeDirectory, "steps.jsonl"), stepLines, Encoding.UTF8);
        File.WriteAllText(
            Path.Combine(episodeDirectory, "summary.json"),
            BuildEpisodeSummaryJson(episodeId, episodeSeed, framesPerEpisode, routeLength),
            Encoding.UTF8
        );

        summaries.Add(new EpisodeSummary
        {
            episodeId = episodeId,
            numSteps = framesPerEpisode,
            seed = episodeSeed,
            routeLength = routeLength
        });
    }

    private string ResolveRunDirectory()
    {
        string root = usePersistentDataPath
            ? Path.Combine(Application.persistentDataPath, outputRoot)
            : outputRoot;
        return Path.Combine(root, runId);
    }

    private List<Vector3> GenerateRoute(System.Random rng)
    {
        List<Vector3> route = new List<Vector3>();
        int lane = 0;
        float z = 1.5f;
        route.Add(new Vector3(0.0f, 1.0f, z));

        for (int i = 1; i < routeWaypoints; i++)
        {
            int direction = rng.NextDouble() < 0.45 ? 0 : (rng.NextDouble() < 0.5 ? -1 : 1);
            lane = Mathf.Clamp(lane + direction, -2, 2);
            float x = lane * laneWidth + RandomRange(rng, -0.45f, 0.45f);
            z += RandomRange(rng, waypointSpacingMin, waypointSpacingMax);
            route.Add(new Vector3(x, 1.0f, z));
        }

        return route;
    }

    private void BuildScene(List<Vector3> route, System.Random rng)
    {
        if (generatedRoot != null)
        {
            DestroyObject(generatedRoot);
        }
        generatedRoot = new GameObject("DDPD Procedural Dataset Scene");
        palette = CreatePalette();

        float length = route[route.Count - 1].z + 8.0f;
        CreateBox("Floor", new Vector3(0.0f, -0.04f, length * 0.5f), new Vector3(corridorHalfWidth * 2.0f, 0.08f, length), palette[0]);
        CreateBox("LeftWall", new Vector3(-corridorHalfWidth, 1.2f, length * 0.5f), new Vector3(0.25f, 2.4f, length), palette[1]);
        CreateBox("RightWall", new Vector3(corridorHalfWidth, 1.2f, length * 0.5f), new Vector3(0.25f, 2.4f, length), palette[1]);

        CreateGrid(length);
        CreateLandmarks(route, rng);
        CreateObstacles(route, rng);
        if (showDebugRouteLine)
        {
            CreateRouteLine(route);
        }

        GameObject lightObject = new GameObject("DatasetLight");
        lightObject.transform.SetParent(generatedRoot.transform, false);
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.1f;
        lightObject.transform.rotation = Quaternion.Euler(55.0f, -35.0f, 0.0f);
    }

    private void BuildAgent(Vector3 start, Vector3 next)
    {
        if (agent != null)
        {
            DestroyObject(agent);
        }
        agent = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        agent.name = "DatasetAgent";
        agent.transform.position = start;
        agent.transform.rotation = Quaternion.LookRotation((next - start).normalized, Vector3.up);
        Renderer renderer = agent.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = CreateMaterial(new Color(0.08f, 0.42f, 1.0f, 1.0f));
        }

        GameObject cameraObject = new GameObject("DatasetCamera");
        cameraObject.transform.SetParent(agent.transform, false);
        cameraObject.transform.localPosition = new Vector3(0.0f, 0.68f, 0.18f);
        cameraObject.transform.localRotation = Quaternion.identity;
        captureCamera = cameraObject.AddComponent<Camera>();
        captureCamera.fieldOfView = 78.0f;
        captureCamera.nearClipPlane = 0.03f;
        captureCamera.farClipPlane = 90.0f;
    }

    private void CreateLandmarks(List<Vector3> route, System.Random rng)
    {
        for (int i = 1; i < route.Count - 1; i++)
        {
            Vector3 point = route[i];
            Material accent = palette[2 + (i % (palette.Length - 2))];
            float side = i % 2 == 0 ? -1.0f : 1.0f;
            float wallX = side * (corridorHalfWidth - 0.18f);
            CreateBox(
                $"WallLandmark_{i:00}",
                new Vector3(wallX, 1.35f, point.z),
                new Vector3(0.08f, 1.35f, RandomRange(rng, 1.2f, 2.4f)),
                accent
            );
            CreateCylinder(
                $"Beacon_{i:00}",
                new Vector3(-side * RandomRange(rng, 2.0f, 4.0f), 0.7f, point.z + RandomRange(rng, -1.4f, 1.4f)),
                new Vector3(0.32f, 0.7f, 0.32f),
                accent
            );
            CreateBox(
                $"FloorCue_{i:00}",
                new Vector3(point.x, 0.022f, point.z),
                new Vector3(RandomRange(rng, 0.7f, 1.5f), 0.025f, RandomRange(rng, 1.2f, 2.0f)),
                accent
            );
        }
    }

    private void CreateObstacles(List<Vector3> route, System.Random rng)
    {
        for (int i = 2; i < route.Count - 2; i++)
        {
            Vector3 point = route[i];
            Vector3 previous = route[i - 1];
            Vector3 next = route[i + 1];
            Vector3 direction = (next - previous).normalized;
            Vector3 right = new Vector3(direction.z, 0.0f, -direction.x);
            float offset = rng.NextDouble() < 0.5 ? -1.6f : 1.6f;
            Vector3 obstaclePosition = point + right * offset;
            obstaclePosition.y = 0.55f;
            CreateBox(
                $"Obstacle_{i:00}",
                obstaclePosition,
                new Vector3(RandomRange(rng, 0.7f, 1.3f), RandomRange(rng, 0.8f, 1.5f), RandomRange(rng, 0.8f, 1.8f)),
                palette[2 + rng.Next(palette.Length - 2)]
            );
        }
    }

    private void CreateGrid(float length)
    {
        Material material = CreateMaterial(new Color(0.55f, 0.85f, 1.0f, 0.9f));
        for (float z = 0.0f; z <= length; z += 1.0f)
        {
            CreateLine($"GridZ_{z:000}", new Vector3(-corridorHalfWidth, 0.015f, z), new Vector3(corridorHalfWidth, 0.015f, z), material, 0.01f);
        }
        for (float x = -corridorHalfWidth; x <= corridorHalfWidth; x += 1.0f)
        {
            CreateLine($"GridX_{x:000}", new Vector3(x, 0.016f, 0.0f), new Vector3(x, 0.016f, length), material, 0.01f);
        }
    }

    private void CreateRouteLine(List<Vector3> route)
    {
        GameObject routeObject = new GameObject("DebugGroundTruthRoute");
        routeObject.transform.SetParent(generatedRoot.transform, false);
        LineRenderer line = routeObject.AddComponent<LineRenderer>();
        ConfigureLine(line, CreateMaterial(new Color(0.2f, 1.0f, 0.25f, 1.0f)), 0.08f);
        line.positionCount = route.Count;
        for (int i = 0; i < route.Count; i++)
        {
            line.SetPosition(i, new Vector3(route[i].x, 0.04f, route[i].z));
        }
    }

    private GameObject CreateBox(string objectName, Vector3 position, Vector3 scale, Material material)
    {
        GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = objectName;
        box.transform.SetParent(generatedRoot.transform, false);
        box.transform.localPosition = position;
        box.transform.localScale = scale;
        Renderer renderer = box.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = material;
        }
        return box;
    }

    private GameObject CreateCylinder(string objectName, Vector3 position, Vector3 scale, Material material)
    {
        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cylinder.name = objectName;
        cylinder.transform.SetParent(generatedRoot.transform, false);
        cylinder.transform.localPosition = position;
        cylinder.transform.localScale = scale;
        Renderer renderer = cylinder.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = material;
        }
        return cylinder;
    }

    private void CreateLine(string objectName, Vector3 start, Vector3 end, Material material, float width)
    {
        GameObject lineObject = new GameObject(objectName);
        lineObject.transform.SetParent(generatedRoot.transform, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        ConfigureLine(line, material, width);
        line.positionCount = 2;
        line.SetPosition(0, start);
        line.SetPosition(1, end);
    }

    private static void ConfigureLine(LineRenderer line, Material material, float width)
    {
        line.useWorldSpace = true;
        line.sharedMaterial = material;
        line.widthMultiplier = width;
        line.numCapVertices = 2;
        line.numCornerVertices = 2;
    }

    private void PoseRoute(List<Vector3> route, float distance, out Vector3 position, out Quaternion rotation)
    {
        float remaining = distance;
        for (int i = 0; i < route.Count - 1; i++)
        {
            Vector3 start = route[i];
            Vector3 end = route[i + 1];
            float segmentLength = Vector3.Distance(start, end);
            if (remaining <= segmentLength)
            {
                float t = segmentLength <= 0.001f ? 0.0f : remaining / segmentLength;
                position = Vector3.Lerp(start, end, t);
                Vector3 direction = (end - start).normalized;
                rotation = Quaternion.LookRotation(direction, Vector3.up);
                return;
            }
            remaining -= segmentLength;
        }

        position = route[route.Count - 1];
        Vector3 finalDirection = (route[route.Count - 1] - route[route.Count - 2]).normalized;
        rotation = Quaternion.LookRotation(finalDirection, Vector3.up);
    }

    private float ComputeRouteLength(List<Vector3> route)
    {
        float total = 0.0f;
        for (int i = 0; i < route.Count - 1; i++)
        {
            total += Vector3.Distance(route[i], route[i + 1]);
        }
        return total;
    }

    private void CapturePng(string path)
    {
        RenderTexture previousTarget = captureCamera.targetTexture;
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture target = RenderTexture.GetTemporary(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
        Texture2D texture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);

        try
        {
            captureCamera.targetTexture = target;
            RenderTexture.active = target;
            captureCamera.Render();
            texture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            texture.Apply();
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllBytes(path, texture.EncodeToPNG());
        }
        finally
        {
            captureCamera.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            RenderTexture.ReleaseTemporary(target);
            DestroyObject(texture);
        }
    }

    private PoseRecord ToPoseRecord(Transform target)
    {
        return new PoseRecord
        {
            x = target.position.z,
            y = target.position.x,
            angle = WrapDegrees(target.eulerAngles.y)
        };
    }

    private string BuildStepJson(
        int step,
        float elapsed,
        string framePath,
        PoseRecord currentPose,
        bool hasPreviousPose,
        PoseRecord previousPose,
        int episodeSeed
    )
    {
        RelativeEgoMotion(currentPose, hasPreviousPose, previousPose, out float forward, out float right, out float dyaw, out float dyawDeg);
        StringBuilder builder = new StringBuilder(512);
        builder.Append("{");
        AppendJsonField(builder, "step", step).Append(",");
        AppendJsonField(builder, "time_sec", elapsed).Append(",");
        AppendJsonField(builder, "frame_path", framePath).Append(",");
        builder.Append("\"pose\":{");
        AppendJsonField(builder, "x", currentPose.x).Append(",");
        AppendJsonField(builder, "y", currentPose.y).Append(",");
        AppendJsonField(builder, "angle", currentPose.angle);
        builder.Append("},");
        builder.Append("\"relative_egomotion_from_prev\":{");
        AppendJsonField(builder, "dx_forward", forward).Append(",");
        AppendJsonField(builder, "dy_right", right).Append(",");
        AppendJsonField(builder, "dyaw", dyaw).Append(",");
        AppendJsonField(builder, "dyaw_deg", dyawDeg);
        builder.Append("},");
        builder.Append("\"metadata\":{");
        AppendJsonField(builder, "generator", "unity_procedural").Append(",");
        AppendJsonField(builder, "seed", episodeSeed);
        builder.Append("}");
        builder.Append("}");
        return builder.ToString();
    }

    private static void RelativeEgoMotion(
        PoseRecord currentPose,
        bool hasPreviousPose,
        PoseRecord previousPose,
        out float forward,
        out float right,
        out float dyaw,
        out float dyawDeg
    )
    {
        if (!hasPreviousPose)
        {
            forward = 0.0f;
            right = 0.0f;
            dyaw = 0.0f;
            dyawDeg = 0.0f;
            return;
        }

        float dx = currentPose.x - previousPose.x;
        float dy = currentPose.y - previousPose.y;
        float yaw = previousPose.angle * Mathf.Deg2Rad;
        forward = Mathf.Cos(yaw) * dx + Mathf.Sin(yaw) * dy;
        right = -Mathf.Sin(yaw) * dx + Mathf.Cos(yaw) * dy;
        dyawDeg = WrapDegrees(currentPose.angle - previousPose.angle);
        dyaw = dyawDeg * Mathf.Deg2Rad;
    }

    private void WriteManifest(string runDirectory)
    {
        StringBuilder builder = new StringBuilder(4096);
        builder.Append("{\n");
        AppendJsonLine(builder, "run_id", runId, 1, true);
        AppendJsonLine(builder, "source_dataset", "unity_procedural", 1, true);
        AppendJsonLine(builder, "env_name", "unity", 1, true);
        AppendJsonLine(builder, "scenario", "procedural_visual_navigation", 1, true);
        AppendJsonLine(builder, "map", "seeded_route_scene", 1, true);
        AppendJsonLine(builder, "fps", sampleFps, 1, true);
        AppendJsonLine(builder, "frame_skip", frameSkip, 1, true);
        AppendJsonLine(builder, "episode_count", summaries.Count, 1, true);
        AppendJsonLine(builder, "max_steps", framesPerEpisode, 1, true);
        AppendJsonLine(builder, "generation_mode", "unity_procedural_dataset", 1, true);
        AppendJsonLine(builder, "policy", "route_follower", 1, true);
        AppendJsonLine(builder, "seed", seed, 1, true);
        builder.Append("  \"enabled_buffers\":{\"rgb\":true,\"depth\":false,\"labels\":false,\"automap\":false},\n");
        builder.Append("  \"episodes\":[\n");
        for (int i = 0; i < summaries.Count; i++)
        {
            EpisodeSummary summary = summaries[i];
            builder.Append("    {");
            AppendJsonField(builder, "episode_id", summary.episodeId).Append(",");
            AppendJsonField(builder, "steps_path", $"episodes/{summary.episodeId}/steps.jsonl").Append(",");
            AppendJsonField(builder, "summary_path", $"episodes/{summary.episodeId}/summary.json");
            builder.Append(i == summaries.Count - 1 ? "}\n" : "},\n");
        }
        builder.Append("  ],\n");
        builder.Append("  \"episode_summaries\":[\n");
        for (int i = 0; i < summaries.Count; i++)
        {
            EpisodeSummary summary = summaries[i];
            builder.Append("    {");
            AppendJsonField(builder, "episode_id", summary.episodeId).Append(",");
            AppendJsonField(builder, "num_steps", summary.numSteps).Append(",");
            AppendJsonField(builder, "seed", summary.seed).Append(",");
            AppendJsonField(builder, "route_length", summary.routeLength);
            builder.Append(i == summaries.Count - 1 ? "}\n" : "},\n");
        }
        builder.Append("  ]\n");
        builder.Append("}\n");
        File.WriteAllText(Path.Combine(runDirectory, "manifest.json"), builder.ToString(), Encoding.UTF8);
    }

    private string BuildEpisodeSummaryJson(string episodeId, int episodeSeed, int steps, float routeLength)
    {
        StringBuilder builder = new StringBuilder(512);
        builder.Append("{\n");
        AppendJsonLine(builder, "episode_id", episodeId, 1, true);
        AppendJsonLine(builder, "num_steps", steps, 1, true);
        AppendJsonLine(builder, "seed", episodeSeed, 1, true);
        AppendJsonLine(builder, "route_length", routeLength, 1, false);
        builder.Append("}\n");
        return builder.ToString();
    }

    private static StringBuilder AppendJsonLine(StringBuilder builder, string key, string value, int indent, bool comma)
    {
        AppendIndent(builder, indent);
        AppendJsonField(builder, key, value);
        builder.Append(comma ? ",\n" : "\n");
        return builder;
    }

    private static StringBuilder AppendJsonLine(StringBuilder builder, string key, int value, int indent, bool comma)
    {
        AppendIndent(builder, indent);
        AppendJsonField(builder, key, value);
        builder.Append(comma ? ",\n" : "\n");
        return builder;
    }

    private static StringBuilder AppendJsonLine(StringBuilder builder, string key, float value, int indent, bool comma)
    {
        AppendIndent(builder, indent);
        AppendJsonField(builder, key, value);
        builder.Append(comma ? ",\n" : "\n");
        return builder;
    }

    private static StringBuilder AppendJsonField(StringBuilder builder, string key, string value)
    {
        builder.Append('"').Append(EscapeJson(key)).Append("\":\"").Append(EscapeJson(value)).Append('"');
        return builder;
    }

    private static StringBuilder AppendJsonField(StringBuilder builder, string key, int value)
    {
        builder.Append('"').Append(EscapeJson(key)).Append("\":").Append(value.ToString(CultureInfo.InvariantCulture));
        return builder;
    }

    private static StringBuilder AppendJsonField(StringBuilder builder, string key, float value)
    {
        builder.Append('"').Append(EscapeJson(key)).Append("\":").Append(value.ToString("R", CultureInfo.InvariantCulture));
        return builder;
    }

    private static void AppendIndent(StringBuilder builder, int indent)
    {
        for (int i = 0; i < indent; i++)
        {
            builder.Append("  ");
        }
    }

    private static string EscapeJson(string value)
    {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private Material[] CreatePalette()
    {
        return new[]
        {
            CreateMaterial(new Color(0.10f, 0.12f, 0.15f, 1.0f)),
            CreateMaterial(new Color(0.18f, 0.22f, 0.28f, 1.0f)),
            CreateMaterial(new Color(1.0f, 0.12f, 0.10f, 1.0f)),
            CreateMaterial(new Color(0.05f, 0.35f, 1.0f, 1.0f)),
            CreateMaterial(new Color(1.0f, 0.78f, 0.06f, 1.0f)),
            CreateMaterial(new Color(0.55f, 0.18f, 1.0f, 1.0f)),
            CreateMaterial(new Color(0.0f, 0.95f, 0.9f, 1.0f)),
            CreateMaterial(new Color(0.95f, 0.45f, 0.05f, 1.0f))
        };
    }

    private static Material CreateMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
        {
            shader = Shader.Find("Unlit/Color");
        }
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }
        Material material = new Material(shader);
        material.color = color;
        return material;
    }

    private static float RandomRange(System.Random rng, float minValue, float maxValue)
    {
        return minValue + (float)rng.NextDouble() * (maxValue - minValue);
    }

    private static float WrapDegrees(float value)
    {
        float wrapped = Mathf.Repeat(value + 180.0f, 360.0f) - 180.0f;
        return wrapped;
    }

    private static void DestroyObject(UnityEngine.Object target)
    {
        if (target == null)
        {
            return;
        }
        if (Application.isPlaying)
        {
            Destroy(target);
        }
        else
        {
            DestroyImmediate(target);
        }
    }
}
