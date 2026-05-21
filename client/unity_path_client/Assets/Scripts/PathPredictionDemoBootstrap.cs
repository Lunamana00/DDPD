using UnityEngine;

public static class PathPredictionDemoBootstrap
{
    private const string DefaultServerUrl = "http://127.0.0.1:8000/predict";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateDemoIfNeeded()
    {
        if (Object.FindFirstObjectByType<PathPredictionClient>() != null)
        {
            return;
        }
        if (Object.FindFirstObjectByType<UnityDatasetGenerator>() != null)
        {
            return;
        }

        GameObject root = new GameObject("DDPD Demo Environment");
        DemoEnvironmentBuilder environment = root.AddComponent<DemoEnvironmentBuilder>();
        environment.Build();

        GameObject agent = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        agent.name = "DemoAgent";
        agent.transform.position = new Vector3(0.0f, 1.0f, 1.5f);
        agent.transform.rotation = Quaternion.identity;
        Renderer agentRenderer = agent.GetComponent<Renderer>();
        if (agentRenderer != null)
        {
            agentRenderer.sharedMaterial = CreateMaterial(new Color(0.1f, 0.55f, 1.0f, 1.0f));
        }
        agent.AddComponent<SimpleAgentController>();

        GameObject cameraObject = new GameObject("AgentCamera");
        cameraObject.transform.SetParent(agent.transform, false);
        cameraObject.transform.localPosition = new Vector3(0.0f, 0.7f, 0.15f);
        cameraObject.transform.localRotation = Quaternion.identity;
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.fieldOfView = 72.0f;
        camera.nearClipPlane = 0.03f;
        camera.farClipPlane = 80.0f;
        Camera.SetupCurrent(camera);

        GameObject lightObject = new GameObject("DirectionalLight");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.2f;
        lightObject.transform.rotation = Quaternion.Euler(55.0f, -35.0f, 0.0f);

        PathPredictionVisualizer visualizer = agent.AddComponent<PathPredictionVisualizer>();
        PathPredictionClient client = agent.AddComponent<PathPredictionClient>();
        client.ConfigureRuntime(agent.transform, camera, visualizer, DefaultServerUrl, true);
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
}
