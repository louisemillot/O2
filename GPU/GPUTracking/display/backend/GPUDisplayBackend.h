// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayBackend.h
/// \author David Rohr

#ifndef GPUDISPLAYBACKEND_H
#define GPUDISPLAYBACKEND_H

#include "GPUCommonDef.h"
#include "../utils/vecpod.h"
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#if defined(GPUCA_DISPLAY_GL3W) && !defined(GPUCA_DISPLAY_OPENGL_CORE)
#define GPUCA_DISPLAY_OPENGL_CORE
#endif

union hmm_mat4;

namespace o2::gpu
{
class GPUDisplay;
class GPUDisplayFrontend;
class GPUDisplayMagneticField;
struct GLfb;
class GPUDisplayBackend
{
  friend GPUDisplay;

 public:
  GPUDisplayBackend();
  virtual ~GPUDisplayBackend();

  virtual int32_t ExtInit() { return 0; };
  virtual bool CoreProfile() { return false; };
  virtual uint32_t DepthBits() = 0;

  typedef std::tuple<uint32_t, uint32_t, int32_t> vboList;

  enum drawType {
    POINTS = 0,
    LINES = 1,
    LINE_STRIP = 2
  };

  enum backendTypes {
    TYPE_INVALID = -1,
    TYPE_OPENGL = 0,
    TYPE_VULKAN = 1
  };

  struct DrawArraysIndirectCommand {
    DrawArraysIndirectCommand(uint32_t a = 0, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0) : count(a), instanceCount(b), first(c), baseInstance(d) {}
    uint32_t count;
    uint32_t instanceCount;

    uint32_t first;
    uint32_t baseInstance;
  };

  struct FontSymbol {
    int32_t size[2];
    int32_t offset[2];
    int32_t advance;
  };

  virtual uint32_t drawVertices(const vboList& v, const drawType t) = 0;
  virtual uint32_t drawField() { return 0; }
  virtual void ActivateColor(std::array<float, 4>& color) = 0;
  virtual void setQuality() {};
  virtual void SetVSync(bool enable) {};
  virtual bool backendNeedRedraw() { return true; }
  virtual void setDepthBuffer() = 0;
  virtual int32_t InitBackendA() = 0;
  virtual void ExitBackendA() = 0;
  int32_t InitBackend();
  void ExitBackend();
  virtual void loadDataToGPU(size_t totalVertizes) = 0;
  virtual void prepareDraw(const hmm_mat4& proj, const hmm_mat4& view, bool requestScreenshot = false, bool toMixBuffer = false, float includeMixImage = 0.f) = 0;
  virtual void finishDraw(bool doScreenshot, bool toMixBuffer = false, float includeMixImage = 0.f) = 0;
  virtual void finishFrame(bool doScreenshot, bool toMixBuffer = false, float includeMixImage = 0.f) = 0;
  virtual void prepareText() = 0;
  virtual void finishText() = 0;
  virtual void pointSizeFactor(float factor) = 0;
  virtual void lineWidthFactor(float factor) = 0;
  backendTypes backendType() const { return mBackendType; }
  const char* backendName() const { return mBackendName; }
  virtual void resizeScene(uint32_t width, uint32_t height) {}
  virtual size_t needMultiVBO() { return 0; }
  virtual void OpenGLPrint(const char* s, float x, float y, float* color, float scale) = 0;
  static GPUDisplayBackend* getBackend(const char* type);
  std::vector<char> getPixels();
  virtual float getYFactor() const { return 1.0f; }
  virtual int32_t getMaxMSAA() const { return 16; }

 protected:
  virtual void addFontSymbol(int32_t symbol, int32_t sizex, int32_t sizey, int32_t offsetx, int32_t offsety, int32_t advance, void* data) = 0;
  virtual void initializeTextDrawing() = 0;

  float getDownsampleFactor(bool screenshot = false);
  void fillIndirectCmdBuffer();
  bool smoothFont();

  GPUDisplay* mDisplay = nullptr;
  std::vector<int32_t> mIndirectSectorOffset;
  vecpod<DrawArraysIndirectCommand> mCmdBuffer;
  bool mFreetypeInitialized = false;
  bool mFrontendCompatTextDraw = false;
  std::vector<char> mScreenshotPixels;

  int32_t mDownsampleFactor = 1;

  uint32_t mRenderWidth = 0;
  uint32_t mRenderHeight = 0;
  uint32_t mScreenWidth = 0;
  uint32_t mScreenHeight = 0;

  backendTypes mBackendType = TYPE_INVALID;
  const char* mBackendName = nullptr;

  std::unique_ptr<GPUDisplayMagneticField> mMagneticFieldVisualization;
};
} // namespace o2::gpu

#endif
