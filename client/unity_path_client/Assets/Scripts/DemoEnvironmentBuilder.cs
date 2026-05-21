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
    [SerializeField] private Color floorColor = new Color(0.12f, 0.14f, 0.16f, 1.0f);
    [SerializeField] private Color wallColor = new Color(0.18f, 0.22f, 0.28f, 1.0f);
    [SerializeField] private Color obstacleColor = new Color(0.82f, 0.42f, 0.12f, 1.0f);
    [SerializeField] private Color routeColor = new Color(0.35f, 1.0f, 0.45f, 1.0f);
    [SerializeField] private Color gridColor = new Color(0.55f, 0.85f, 1.0f, 0.9f);

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
        Material redMaterial = CreateMaterial(new Color(1.0f, 0.12f, 0.10f, 1.0f));
        Material blueMaterial = CreateMaterial(new Color(0.05f, 0.35f, 1.0f, 1.0f));
        Material yellowMaterial = CreateMaterial(new Color(1.0f, 0.78f, 0.06f, 1.0f));
        Material purpleMaterial = CreateMaterial(new Color(0.55f, 0.18f, 1.0f, 1.0f));
        Material cyanMaterial = CreateMaterial(new Color(0.0f, 0.95f, 0.9f, 1.0f));
        Material darkMaterial = CreateMaterial(new Color(0.04f, 0.05f, 0.07f, 1.0f));

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

        CreateVisionCues(redMaterial, blueMaterial, yellowMaterial, purpleMaterial, cyanMaterial, darkMaterial);
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

    private GameObject CreatePanel(
        string objectName,
        Vector3 localPosition,
        Vector3 localScale,
        Material material,
        float yRotation = 0.0f
    )
    {
        GameObject panel = CreateBox(objectName, localPosition, localScale, material);
        panel.transform.localRotation = Quaternion.Euler(0.0f, yRotation, 0.0f);
        return panel;
    }

    private void CreateVisionCues(
        Material redMaterial,
        Material blueMaterial,
        Material yellowMaterial,
        Material purpleMaterial,
        Material cyanMaterial,
        Material darkMaterial
    )
    {
        CreateFloorBands(redMaterial, blueMaterial, yellowMaterial, purpleMaterial, cyanMaterial);
        CreateWallLandmarks(redMaterial, blueMaterial, yellowMaterial, purpleMaterial, cyanMaterial, darkMaterial);
        CreateDecisionGates(redMaterial, blueMaterial, yellowMaterial, cyanMaterial, darkMaterial);
        CreateObjectCueClusters(redMaterial, blueMaterial, yellowMaterial, purpleMaterial, cyanMaterial);
        CreateOccludersAndOpenings(darkMaterial, yellowMaterial, cyanMaterial);
    }

    private void CreateFloorBands(
        Material redMaterial,
        Material blueMaterial,
        Material yellowMaterial,
        Material purpleMaterial,
        Material cyanMaterial
    )
    {
        CreatePanel("BlueLaneCue", new Vector3(-2.2f, 0.018f, 6.0f), new Vector3(0.42f, 0.025f, 7.0f), blueMaterial);
        CreatePanel("YellowLaneCue", new Vector3(2.1f, 0.019f, 13.5f), new Vector3(0.42f, 0.025f, 7.0f), yellowMaterial);
        CreatePanel("PurpleLaneCue", new Vector3(-1.5f, 0.020f, 21.0f), new Vector3(0.42f, 0.025f, 7.0f), purpleMaterial);
        CreatePanel("CyanLaneCue", new Vector3(1.6f, 0.021f, 28.0f), new Vector3(0.42f, 0.025f, 6.0f), cyanMaterial);

        for (int i = 0; i < 8; i++)
        {
            Material material = i % 2 == 0 ? redMaterial : blueMaterial;
            float x = -3.5f + i;
            CreatePanel(
                $"StartFloorColorTile{i:00}",
                new Vector3(x, 0.026f, 2.4f),
                new Vector3(0.78f, 0.025f, 0.78f),
                material
            );
        }
    }

    private void CreateWallLandmarks(
        Material redMaterial,
        Material blueMaterial,
        Material yellowMaterial,
        Material purpleMaterial,
        Material cyanMaterial,
        Material darkMaterial
    )
    {
        float leftX = -corridorWidth * 0.5f + 0.16f;
        float rightX = corridorWidth * 0.5f - 0.16f;
        CreatePanel("LeftRedWallPanel", new Vector3(leftX, 1.35f, 5.0f), new Vector3(0.08f, 1.4f, 2.6f), redMaterial);
        CreatePanel("RightBlueWallPanel", new Vector3(rightX, 1.35f, 8.5f), new Vector3(0.08f, 1.4f, 2.6f), blueMaterial);
        CreatePanel("LeftYellowWallPanel", new Vector3(leftX, 1.35f, 14.0f), new Vector3(0.08f, 1.4f, 2.6f), yellowMaterial);
        CreatePanel("RightPurpleWallPanel", new Vector3(rightX, 1.35f, 19.0f), new Vector3(0.08f, 1.4f, 2.6f), purpleMaterial);
        CreatePanel("LeftCyanWallPanel", new Vector3(leftX, 1.35f, 26.0f), new Vector3(0.08f, 1.4f, 2.6f), cyanMaterial);

        for (int i = 0; i < 10; i++)
        {
            Material material = i % 2 == 0 ? darkMaterial : cyanMaterial;
            CreatePanel(
                $"RightBarcodePanel{i:00}",
                new Vector3(rightX - 0.01f, 1.7f, 24.5f + i * 0.28f),
                new Vector3(0.05f, 0.9f, 0.08f),
                material
            );
        }
    }

    private void CreateDecisionGates(
        Material redMaterial,
        Material blueMaterial,
        Material yellowMaterial,
        Material cyanMaterial,
        Material darkMaterial
    )
    {
        CreateGate("BlueGate", 10.0f, blueMaterial, darkMaterial, -1.6f);
        CreateGate("YellowGate", 18.0f, yellowMaterial, darkMaterial, 1.4f);
        CreateGate("CyanGate", 27.0f, cyanMaterial, darkMaterial, -0.8f);

        CreateArrow("LeftTurnCue", new Vector3(-2.6f, 1.2f, 10.0f), redMaterial, -25.0f);
        CreateArrow("RightTurnCue", new Vector3(2.6f, 1.2f, 18.0f), yellowMaterial, 25.0f);
        CreateArrow("CenterCue", new Vector3(0.0f, 1.2f, 27.0f), cyanMaterial, 0.0f);
    }

    private void CreateGate(string objectName, float z, Material accentMaterial, Material frameMaterial, float gapCenterX)
    {
        GameObject gate = new GameObject(objectName);
        gate.transform.SetParent(transform, false);
        CreateBox($"{objectName}Top", new Vector3(0.0f, 2.25f, z), new Vector3(corridorWidth, 0.22f, 0.25f), frameMaterial)
            .transform.SetParent(gate.transform, true);
        CreateBox($"{objectName}LeftPost", new Vector3(gapCenterX - 1.0f, 1.1f, z), new Vector3(0.18f, 2.1f, 0.25f), accentMaterial)
            .transform.SetParent(gate.transform, true);
        CreateBox($"{objectName}RightPost", new Vector3(gapCenterX + 1.0f, 1.1f, z), new Vector3(0.18f, 2.1f, 0.25f), accentMaterial)
            .transform.SetParent(gate.transform, true);
    }

    private void CreateArrow(string objectName, Vector3 localPosition, Material material, float yRotation)
    {
        GameObject arrow = new GameObject(objectName);
        arrow.transform.SetParent(transform, false);
        arrow.transform.localPosition = localPosition;
        arrow.transform.localRotation = Quaternion.Euler(0.0f, yRotation, 0.0f);

        GameObject shaft = CreateBox($"{objectName}Shaft", Vector3.zero, new Vector3(0.18f, 0.18f, 1.1f), material);
        shaft.transform.SetParent(arrow.transform, false);
        shaft.transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);

        GameObject head = CreateBox($"{objectName}Head", Vector3.zero, new Vector3(0.62f, 0.2f, 0.42f), material);
        head.transform.SetParent(arrow.transform, false);
        head.transform.localPosition = new Vector3(0.0f, 0.0f, 0.72f);
        head.transform.localRotation = Quaternion.Euler(0.0f, 45.0f, 0.0f);
    }

    private void CreateObjectCueClusters(
        Material redMaterial,
        Material blueMaterial,
        Material yellowMaterial,
        Material purpleMaterial,
        Material cyanMaterial
    )
    {
        CreateCylinderCue("BlueBeacon", new Vector3(-3.2f, 0.75f, 6.5f), blueMaterial);
        CreateCylinderCue("YellowBeacon", new Vector3(3.1f, 0.75f, 13.0f), yellowMaterial);
        CreateCylinderCue("PurpleBeacon", new Vector3(-2.9f, 0.75f, 20.5f), purpleMaterial);
        CreateCylinderCue("CyanBeacon", new Vector3(2.8f, 0.75f, 29.0f), cyanMaterial);

        CreateBox("RedCrateA", new Vector3(1.1f, 0.45f, 7.5f), new Vector3(0.8f, 0.9f, 0.8f), redMaterial);
        CreateBox("BlueCrateB", new Vector3(-1.0f, 0.45f, 16.5f), new Vector3(0.8f, 0.9f, 0.8f), blueMaterial);
        CreateBox("YellowCrateC", new Vector3(1.1f, 0.45f, 25.0f), new Vector3(0.8f, 0.9f, 0.8f), yellowMaterial);
    }

    private void CreateCylinderCue(string objectName, Vector3 localPosition, Material material)
    {
        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cylinder.name = objectName;
        cylinder.transform.SetParent(transform, false);
        cylinder.transform.localPosition = localPosition;
        cylinder.transform.localScale = new Vector3(0.36f, 0.75f, 0.36f);
        Renderer renderer = cylinder.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.sharedMaterial = material;
        }
    }

    private void CreateOccludersAndOpenings(Material darkMaterial, Material yellowMaterial, Material cyanMaterial)
    {
        CreatePanel("NearOccluderLeft", new Vector3(-0.6f, 0.95f, 11.8f), new Vector3(0.34f, 1.9f, 2.4f), darkMaterial, 12.0f);
        CreatePanel("NearOccluderRight", new Vector3(0.9f, 0.95f, 20.5f), new Vector3(0.34f, 1.9f, 2.2f), darkMaterial, -14.0f);
        CreatePanel("YellowDecisionPad", new Vector3(2.6f, 0.035f, 18.0f), new Vector3(1.6f, 0.025f, 1.6f), yellowMaterial);
        CreatePanel("CyanDecisionPad", new Vector3(-1.2f, 0.036f, 27.0f), new Vector3(1.6f, 0.025f, 1.6f), cyanMaterial);
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
