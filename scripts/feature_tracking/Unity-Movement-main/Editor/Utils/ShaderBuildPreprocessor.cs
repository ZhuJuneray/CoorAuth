// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEditor.Build;
using UnityEditor.Rendering;

namespace Oculus.Movement.Utils
{
    /// <summary>
    /// Process which shader variants get included in a build.
    /// </summary>
    public class ShaderBuildPreprocessor : IPreprocessShaders
    {
        /// <summary>
        /// Used to set when Unity calls this shader preprocessor.
        /// </summary>
        public int callbackOrder => 0;

        /// <summary>
        /// Callback before each shader is compiled.
        /// </summary>
        /// <param name="shader">The shader about to be compiled.</param>
        /// <param name="snippet">Shader details.</param>
        /// <param name="data">List of shader variants.</param>
        public void OnProcessShader(Shader shader, ShaderSnippetData snippet, IList<ShaderCompilerData> data)
        {
            // Exclude URP shader variants from built-in pipeline.
            if (GraphicsSettings.renderPipelineAsset == null && snippet.passType == PassType.ScriptableRenderPipeline)
            {
                for (int i = 0; i < data.Count; ++i)
                {
                    data.RemoveAt(i);
                    i--;
                }
            }
        }
    }
}
