using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class DDPDUnityDatasetBatch
{
    private const string DefaultRunId = "unity_procedural_001";
    private const int DefaultEpisodes = 4;
    private const int DefaultFramesPerEpisode = 180;
    private const int DefaultCaptureSize = 128;

    private sealed class GenerationOptions
    {
        public string RunId = DefaultRunId;
        public string RawRoot;
        public int Episodes = DefaultEpisodes;
        public int FramesPerEpisode = DefaultFramesPerEpisode;
        public int CaptureSize = DefaultCaptureSize;
    }

    [MenuItem("DDPD/Generate Unity Raw Dataset")]
    public static void GenerateDefaultRawDataset()
    {
        GenerationOptions options = ResolveOptions();
        string outputPath = GenerateRawDataset(options);
        Debug.Log($"Generated DDPD raw dataset: {outputPath}");
    }

    public static void GenerateDefaultAndQuit()
    {
        try
        {
            GenerateDefaultRawDataset();
            EditorApplication.Exit(0);
        }
        catch (System.Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static string GenerateRawDataset(GenerationOptions options)
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        GameObject generatorObject = new GameObject("DDPD Unity Dataset Generator");
        UnityDatasetGenerator generator = generatorObject.AddComponent<UnityDatasetGenerator>();
        generator.ConfigureOutput(
            options.RunId,
            options.RawRoot,
            false,
            options.Episodes,
            options.FramesPerEpisode,
            options.CaptureSize,
            options.CaptureSize
        );

        string outputPath = generator.GenerateDatasetBlocking();
        AssetDatabase.Refresh();
        return outputPath;
    }

    private static GenerationOptions ResolveOptions()
    {
        string repoRoot = ResolveRepoRoot();
        GenerationOptions options = new GenerationOptions
        {
            RawRoot = Path.Combine(repoRoot, "data", "wit_vz", "raw")
        };

        string[] args = Environment.GetCommandLineArgs();
        options.RunId = GetStringArg(args, "--ddpd-run-id", options.RunId);
        options.RawRoot = ToAbsolutePath(GetStringArg(args, "--ddpd-raw-root", options.RawRoot), repoRoot);
        options.Episodes = GetIntArg(args, "--ddpd-episodes", options.Episodes);
        options.FramesPerEpisode = GetIntArg(args, "--ddpd-frames-per-episode", options.FramesPerEpisode);
        options.CaptureSize = GetIntArg(args, "--ddpd-capture-size", options.CaptureSize);
        return options;
    }

    private static string GetStringArg(string[] args, string name, string fallback)
    {
        string prefix = name + "=";
        for (int index = 0; index < args.Length; index++)
        {
            string arg = args[index];
            if (arg.StartsWith(prefix, StringComparison.Ordinal))
            {
                return arg.Substring(prefix.Length);
            }
            if (arg == name && index + 1 < args.Length)
            {
                return args[index + 1];
            }
        }
        return fallback;
    }

    private static int GetIntArg(string[] args, string name, int fallback)
    {
        string value = GetStringArg(args, name, null);
        if (int.TryParse(value, out int parsed))
        {
            return Mathf.Max(1, parsed);
        }
        return fallback;
    }

    private static string ToAbsolutePath(string path, string baseDirectory)
    {
        if (Path.IsPathRooted(path))
        {
            return Path.GetFullPath(path);
        }
        return Path.GetFullPath(Path.Combine(baseDirectory, path));
    }

    private static string ResolveRepoRoot()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        DirectoryInfo projectInfo = new DirectoryInfo(projectRoot);
        DirectoryInfo clientInfo = projectInfo.Parent;
        if (clientInfo == null || clientInfo.Parent == null)
        {
            return projectRoot;
        }
        return clientInfo.Parent.FullName;
    }
}
