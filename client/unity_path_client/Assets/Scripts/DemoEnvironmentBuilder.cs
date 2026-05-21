using UnityEngine;

public class DemoEnvironmentBuilder : MonoBehaviour
{
    [Header("Layout")]
    [SerializeField] private float corridorLength = 36.0f;
    [SerializeField] private float corridorWidth = 9.0f;
    [SerializeField] private float wallHeight = 2.4f;
    [SerializeField] private float wallThickness = 0.25f;
    [SerializeField] private float floorThickness = 0.08f;
    [SerializeField] private float gridSpacing = 1.0f;

    [Header("Colors")]
    [SerializeField] private Color floorColor = new Color(0.16f, 0.18f, 0.2f, 1.0f);
    [SerializeField] private Color wallColor = new Color(0.35f, 0.38f, 0.42f, 1.0f);
    [SerializeField] private Color obstacleColor = new Color(0.22f, 0.28f, 0.34f, 1.0f);
    [SerializeField] private Color routeColor = new Color(0.35f, 1.0f, 0.45f, 1.0f);
    [SerializeField] private Color gridColor = new Color(1.0f, 1.0f, 1.0f, 0.22f);

    private bool built;

    private void Start()
    {
        Build();
    }

    public void Build()
    {
        if (built)
        {
            return;
        }
        built = true;

        Material floorMaterial = CreateMaterial(floorColor);
        Material wallMaterial = CreateMaterial(wallColor);
        Material obstacleMaterial = CreateMaterial(obstacleColor);
        Material routeMaterial = CreateMaterial(routeColor);
        Material gridMaterial = CreateMaterial(gridColor);

        CreateBox(
            "Floor",
            new Vector3(0.0f, -floorThickness * 0.5f, corridorLength * 0.5f),
            new Vector3(corridorWidth, floorThickness, corridorLength),
            floorMaterial
        );
        CreateBox(
            "LeftWall",
            new Vector3(-corridorWidth * 0.5f, wallHeight * 0.5f, corridorLength * 0.5f),
            new Vector3(wallThickness, wallHeight, corridorLength),
            wallMaterial
        );
        CreateBox(
            "RightWall",
            new Vector3(corridorWidth * 0.5f, wallHeight * 0.5f, corridorLength * 0.5f),
            new Vector3(wallThickness, wallHeight, corridorLength),
            wallMaterial
        );
        CreateBox(
            "BackWall",
            new Vector3(0.0f, wallHeight * 0.5f, corridorLength),
            new Vector3(corridorWidth, wallHeight, wallThickness),
            wallMaterial
        );

        CreateObstacle("ObstacleA", new Vector3(-2.4f, 0.55f, 8.0f), new Vector3(1.2f, 1.1f, 1.8f), obstacleMaterial);
        CreateObstacle("ObstacleB", new Vector3(2.5f, 0.55f, 15.0f), new Vector3(1.4f, 1.1f, 2.0f), obstacleMaterial);
        CreateObstacle("ObstacleC", new Vector3(-1.8f, 0.55f, 23.0f), new Vector3(1.2f, 1.1f, 1.4f), obstacleMaterial);

        CreateGrid(gridMaterial);
        CreateReferenceRoute(routeMaterial);
    }

    private void CreateObstacle(string objectName, Vector3 localPosition, Vector3 localScale, Material material)
    {
        CreateBox(objectName, localPosition, localScale, material);
    }

    private GameObject CreateBox(string objectName, Vector3 localPosition, Vector3 localScale, Material material)
    {
        GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
        box.name = objectName;
        box.transform.SetParent(transform, false);
        box.transform.localPosition = localPosition;
        box.transform.localScale = localScale;
        Renderer renderer = box.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = material;
        }
        return box;
    }

    private void CreateGrid(Material material)
    {
        GameObject gridRoot = new GameObject("FloorGrid");
        gridRoot.transform.SetParent(transform, false);
        float halfWidth = corridorWidth * 0.5f;
        int verticalLines = Mathf.FloorToInt(corridorLength / gridSpacing);
        int horizontalLines = Mathf.FloorToInt(corridorWidth / gridSpacing);

        for (int i = 0; i <= verticalLines; i++)
        {
            float z = i * gridSpacing;
            CreateLine(
                gridRoot.transform,
                "GridZ",
                new Vector3(-halfWidth, 0.012f, z),
                new Vector3(halfWidth, 0.012f, z),
                material,
                0.012f
            );
        }

        for (int i = 0; i <= horizontalLines; i++)
        {
            float x = -halfWidth + i * gridSpacing;
            CreateLine(
                gridRoot.transform,
                "GridX",
                new Vector3(x, 0.013f, 0.0f),
                new Vector3(x, 0.013f, corridorLength),
                material,
                0.012f
            );
        }
    }

    private void CreateReferenceRoute(Material material)
    {
        GameObject route = new GameObject("ReferenceRoute");
        route.transform.SetParent(transform, false);
        LineRenderer line = route.AddComponent<LineRenderer>();
        ConfigureLine(line, material, 0.06f);

        Vector3[] points =
        {
            new Vector3(0.0f, 0.04f, 1.0f),
            new Vector3(0.0f, 0.04f, 6.0f),
            new Vector3(1.8f, 0.04f, 11.0f),
            new Vector3(1.2f, 0.04f, 17.0f),
            new Vector3(-1.6f, 0.04f, 24.0f),
            new Vector3(0.0f, 0.04f, 32.0f)
        };
        line.positionCount = points.Length;
        line.SetPositions(points);
    }

    private void CreateLine(
        Transform parent,
        string objectName,
        Vector3 start,
        Vector3 end,
        Material material,
        float width
    )
    {
        GameObject lineObject = new GameObject(objectName);
        lineObject.transform.SetParent(parent, false);
        LineRenderer line = lineObject.AddComponent<LineRenderer>();
        ConfigureLine(line, material, width);
        line.positionCount = 2;
        line.SetPosition(0, start);
        line.SetPosition(1, end);
    }

    private static void ConfigureLine(LineRenderer line, Material material, float width)
    {
        line.useWorldSpace = false;
        line.sharedMaterial = material;
        line.widthMultiplier = width;
        line.numCapVertices = 2;
        line.numCornerVertices = 2;
    }

    private static Material CreateMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
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
