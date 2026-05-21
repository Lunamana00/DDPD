using System.Collections.Generic;
using UnityEngine;

public class PathPredictionVisualizer : MonoBehaviour
{
    [Header("Line")]
    [SerializeField] private LineRenderer pathLine;
    [SerializeField] private Color lineColor = new Color(0.1f, 0.8f, 1.0f, 1.0f);
    [SerializeField] private float lineWidth = 0.08f;

    [Header("Markers")]
    [SerializeField] private bool showWaypoints = true;
    [SerializeField] private GameObject waypointPrefab;
    [SerializeField] private float waypointRadius = 0.12f;
    [SerializeField] private Color waypointColor = new Color(1.0f, 0.8f, 0.15f, 1.0f);
    [SerializeField] private Color finalWaypointColor = new Color(1.0f, 0.25f, 0.2f, 1.0f);

    [Header("Coordinates")]
    [SerializeField] private Transform agentRoot;
    [SerializeField] private float pathHeight = 0.05f;
    [SerializeField] private float pathScale = 1.0f;
    [SerializeField] private bool drawLocalAxes = true;
    [SerializeField] private float axisLength = 1.0f;

    private readonly List<GameObject> waypointMarkers = new List<GameObject>();
    private Material lineMaterial;
    private Material waypointMaterial;
    private Material finalWaypointMaterial;

    private void Reset()
    {
        agentRoot = transform;
        pathLine = GetComponent<LineRenderer>();
    }

    private void Awake()
    {
        EnsureLineRenderer();
        EnsureMaterials();
    }

    public void Configure(Transform root, float scale, float height)
    {
        agentRoot = root != null ? root : transform;
        pathScale = scale;
        pathHeight = height;
    }

    public void Render(PathPredictionResponse response)
    {
        if (!response.HasPath())
        {
            Clear();
            return;
        }

        EnsureLineRenderer();
        EnsureMaterials();

        pathLine.positionCount = response.path.Length;
        for (int i = 0; i < response.path.Length; i++)
        {
            Vector3 world = PathPointToWorld(response.path[i]);
            pathLine.SetPosition(i, world);
        }

        if (showWaypoints)
        {
            RenderWaypoints(response.path);
        }
        else
        {
            HideWaypoints();
        }
    }

    public void Clear()
    {
        if (pathLine != null)
        {
            pathLine.positionCount = 0;
        }
        HideWaypoints();
    }

    private Vector3 PathPointToWorld(PathPoint point)
    {
        Transform root = agentRoot != null ? agentRoot : transform;
        Vector3 local = new Vector3(
            point.right * pathScale,
            pathHeight,
            point.forward * pathScale
        );
        return root.TransformPoint(local);
    }

    private void RenderWaypoints(PathPoint[] points)
    {
        EnsureMarkerCount(points.Length);
        for (int i = 0; i < waypointMarkers.Count; i++)
        {
            bool active = i < points.Length;
            GameObject marker = waypointMarkers[i];
            marker.SetActive(active);
            if (!active)
            {
                continue;
            }

            marker.transform.position = PathPointToWorld(points[i]);
            marker.transform.localScale = Vector3.one * waypointRadius;
            Renderer renderer = marker.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = i == points.Length - 1 ? finalWaypointMaterial : waypointMaterial;
            }
        }
    }

    private void EnsureMarkerCount(int count)
    {
        while (waypointMarkers.Count < count)
        {
            GameObject marker = CreateWaypointMarker();
            marker.transform.SetParent(transform, false);
            waypointMarkers.Add(marker);
        }
    }

    private GameObject CreateWaypointMarker()
    {
        if (waypointPrefab != null)
        {
            return Instantiate(waypointPrefab);
        }
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = "PredictedPathWaypoint";
        Collider collider = marker.GetComponent<Collider>();
        if (collider != null)
        {
            Destroy(collider);
        }
        return marker;
    }

    private void HideWaypoints()
    {
        foreach (GameObject marker in waypointMarkers)
        {
            if (marker != null)
            {
                marker.SetActive(false);
            }
        }
    }

    private void EnsureLineRenderer()
    {
        if (pathLine == null)
        {
            pathLine = GetComponent<LineRenderer>();
        }
        if (pathLine == null)
        {
            pathLine = gameObject.AddComponent<LineRenderer>();
        }

        pathLine.useWorldSpace = true;
        pathLine.widthMultiplier = lineWidth;
        pathLine.numCornerVertices = 4;
        pathLine.numCapVertices = 4;
    }

    private void EnsureMaterials()
    {
        if (lineMaterial == null)
        {
            lineMaterial = CreateMaterial(lineColor);
            if (pathLine != null)
            {
                pathLine.sharedMaterial = lineMaterial;
                pathLine.startColor = lineColor;
                pathLine.endColor = lineColor;
            }
        }
        if (waypointMaterial == null)
        {
            waypointMaterial = CreateMaterial(waypointColor);
        }
        if (finalWaypointMaterial == null)
        {
            finalWaypointMaterial = CreateMaterial(finalWaypointColor);
        }
    }

    private static Material CreateMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }
        Material material = new Material(shader);
        material.color = color;
        return material;
    }

    private void OnDrawGizmosSelected()
    {
        if (!drawLocalAxes)
        {
            return;
        }

        Transform root = agentRoot != null ? agentRoot : transform;
        Vector3 origin = root.position + Vector3.up * pathHeight;
        Gizmos.color = Color.green;
        Gizmos.DrawLine(origin, origin + root.forward * axisLength);
        Gizmos.color = Color.red;
        Gizmos.DrawLine(origin, origin + root.right * axisLength);
    }
}
