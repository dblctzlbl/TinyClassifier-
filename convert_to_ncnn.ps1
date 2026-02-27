# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 望着天空的眼睛 <a15234181830@163.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

param(
    [string]$OnnxPath = "artifacts/tiny_classifier_96.onnx",
    [string]$OutParam = "artifacts/tiny_classifier_96.ncnn.param",
    [string]$OutBin = "artifacts/tiny_classifier_96.ncnn.bin",
    [string]$OutParamOpt = "artifacts/tiny_classifier_96.opt.param",
    [string]$OutBinOpt = "artifacts/tiny_classifier_96.opt.bin"
)

function Find-Tool($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $localHits = Get-ChildItem -Path . -Recurse -Filter $name -ErrorAction SilentlyContinue
    if ($localHits) {
        if ($env:PROCESSOR_ARCHITECTURE -eq "AMD64") {
            $preferX64 = $localHits | Where-Object { $_.FullName -match "\\x64\\" } | Select-Object -First 1
            if ($preferX64) {
                return $preferX64.FullName
            }
        }
        if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
            $preferArm64 = $localHits | Where-Object { $_.FullName -match "\\arm64\\" } | Select-Object -First 1
            if ($preferArm64) {
                return $preferArm64.FullName
            }
        }
        return ($localHits | Select-Object -First 1).FullName
    }
    return $null
}

$onnx2ncnnPath = Find-Tool "onnx2ncnn.exe"
$pnnxPath = Find-Tool "pnnx.exe"

if ($onnx2ncnnPath) {
    & $onnx2ncnnPath $OnnxPath $OutParam $OutBin
    if ($LASTEXITCODE -ne 0) {
        Write-Error "onnx2ncnn conversion failed."
        exit $LASTEXITCODE
    }
} elseif ($pnnxPath) {
    & $pnnxPath $OnnxPath inputshape=[1,3,96,96] fp16=0
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pnnx conversion failed."
        exit $LASTEXITCODE
    }

    $generatedParam = [System.IO.Path]::ChangeExtension($OnnxPath, ".ncnn.param")
    $generatedBin = [System.IO.Path]::ChangeExtension($OnnxPath, ".ncnn.bin")
    Move-Item -Force $generatedParam $OutParam
    Move-Item -Force $generatedBin $OutBin
} else {
    Write-Error "Neither onnx2ncnn.exe nor pnnx.exe was found."
    exit 1
}

$ncnnoptPath = Find-Tool "ncnnoptimize.exe"
if ($ncnnoptPath) {
    try {
        & $ncnnoptPath $OutParam $OutBin $OutParamOpt $OutBinOpt 0
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Optimized model generated: $OutParamOpt, $OutBinOpt"
        }
    } catch {
        Write-Host "ncnnoptimize execution failed, but base ncnn model is generated."
    }
}

Write-Host "Conversion completed: $OutParam, $OutBin"
