/**
 * HDRI Atmospheric Scattering Renderer
 * 
 * Generates physically-based HDR environment maps with realistic atmospheric 
 * scattering effects. Implements unified Rayleigh and Mie scattering with 
 * atmospheric boundary layer effects, volumetric clouds, and dynamic sun coloring.
 * 
 * Key Features:
 * - Unified atmospheric scattering (Rayleigh + Mie + dust particles)
 * - Physically-accurate sun disk rendering with atmospheric color filtering
 * - Volumetric cloud simulation using Perlin noise
 * - Atmospheric boundary layer for enhanced sunset/sunrise effects
 * - Multiple scattering approximation for realistic sky luminance
 * - Pre-computed lookup tables for phase functions and density calculations
 * - Configurable atmospheric conditions for diverse weather scenarios
 * 
 * The renderer attempts to produces bright, vibrant sunsets with proper color temperature 
 * progression from white (clean atmosphere) through yellow/orange to deep red 
 * (heavy atmospheric content) while maintaining realistic brightness levels.
 * 
 * Output: 32-bit RGBE HDR images suitable for environment mapping
 * 
 */

#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iostream>

// Feature flags for conditional compilation
#define ENABLE_UNIFIED_SCATTERING 1
#define ENABLE_CLOUDS 1
#define ENABLE_WATER 1

/**
 * SIMD-aligned 3D vector structure
 * 16-byte alignment for potential compiler optimizations
 */
struct alignas(16) Vec3 {
    float x_, y_, z_;
    float padding_;  // Ensures 16-byte alignment
    
    Vec3(float x = 0, float y = 0, float z = 0) : x_(x), y_(y), z_(z), padding_(0) {}
    
    // Vector arithmetic operators
    Vec3 operator+(const Vec3& other) const { 
        return Vec3(x_ + other.x_, y_ + other.y_, z_ + other.z_); 
    }
    Vec3 operator-(const Vec3& other) const { 
        return Vec3(x_ - other.x_, y_ - other.y_, z_ - other.z_); 
    }
    Vec3 operator*(float scalar) const { 
        return Vec3(x_ * scalar, y_ * scalar, z_ * scalar); 
    }
    Vec3 operator*(const Vec3& other) const { 
        return Vec3(x_ * other.x_, y_ * other.y_, z_ * other.z_); 
    }
    
    inline float
    dot(const Vec3& other) const { 
        return x_ * other.x_ + y_ * other.y_ + z_ * other.z_; 
    }
    
    inline float
    length() const { 
        return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_); 
    }
    
    inline Vec3
    normalize() const {
        float len = length();
        return len > 0 ? Vec3(x_ / len, y_ / len, z_ / len) : Vec3(0, 0, 0);
    }
};

/**
 * Result structure for sun color calculations
 * Contains color multiplier, visibility, and atmospheric content metrics
 */
struct SunColorResult {
    Vec3 sunColorMultiplier_;    // RGB multiplier for sun color temperature
    float sunVisibility_;        // Sun visibility factor (0.0 = blocked, 1.0 = clear)
    float totalContent_;         // Total atmospheric content measurement
};

/**
 * Configuration for atmospheric content-based sun coloring
 * Controls how atmospheric particles affect sun color and visibility
 */
struct AtmosphereContentSettings {
    // Content contribution factors
    float ablReddeningFactor_ = 0.05f;      // Atmospheric boundary layer contribution
    float cloudReddeningFactor_ = 0.3f;     // Cloud particle contribution
    float maxReddening_ = 15.0f;            // Maximum reddening before saturation
    
    // Adaptive sampling parameters
    int minSunSamples_ = 8;                 // Minimum ray marching samples for high sun
    int maxSunSamples_ = 24;                // Maximum ray marching samples for low sun
    float lowSunThreshold_ = 0.2f;          // Sun elevation threshold for adaptive sampling
    
    // Color temperature mapping
    bool useExponentialMapping_ = false;    // Use exponential vs linear color mapping
    float blueExtinctionRate_ = 0.06f;      // Blue light extinction rate
    float greenExtinctionRate_ = 0.03f;     // Green light extinction rate
    float redExtinctionRate_ = 0.01f;       // Red light extinction rate (least affected)
    
    // Visibility calculation
    float totalBlockageThreshold_ = 100.0f; // Threshold for complete sun blockage
    float visibilityFalloffRate_ = 0.01f;   // Rate of visibility decrease with content
};

/**
 * Comprehensive atmospheric rendering settings
 * Contains all parameters for atmospheric scattering simulation
 */
struct AtmosphereSettings {
    // Solar parameters
    float sunElevationDegrees_ = 2.7f;      // Sun elevation above horizon (degrees)
    float sunAzimuthDegrees_ = 45.0f;       // Sun azimuth direction (degrees)
    float sunIntensity_ = 20.0f;            // Solar irradiance multiplier
    
    // Planetary and atmospheric geometry
    float planetRadius_ = 6371000.0f;       // Planet radius in meters (Earth)
    float atmosphereHeight_ = 80000.0f;     // Atmosphere thickness in meters
    float rayleighScaleHeight_ = 8000.0f;   // Rayleigh scattering scale height
    float mieScaleHeight_ = 1200.0f;        // Mie scattering scale height
    
    // Scattering coefficients (per meter at sea level)
    float rayleighCoeffR_ = 5.8e-6f;        // Rayleigh coefficient for red wavelength
    float rayleighCoeffG_ = 13.5e-6f;       // Rayleigh coefficient for green wavelength
    float rayleighCoeffB_ = 33.1e-6f;       // Rayleigh coefficient for blue wavelength
    float mieCoefficient_ = 21e-6f;         // Mie scattering coefficient
    float mieG_ = 0.95f;                    // Mie phase function asymmetry parameter
    
    // Volumetric cloud parameters
    float cloudDensityMultiplier_ = 0.1f;   // Overall cloud density scaling
    float cloudAltitudeMin_ = 1000.0f;      // Minimum cloud altitude (meters)
    float cloudAltitudeMax_ = 8000.0f;      // Maximum cloud altitude (meters)
    float cloudExtinctionCoeff_ = 0.01f;    // Cloud light extinction coefficient
    float cloudScatteringCoeff_ = 0.009f;   // Cloud light scattering coefficient
    float cloudScale_ = 0.0001f;            // Perlin noise scale for cloud generation
    
    // Ray marching parameters
    int primarySamples_ = 8;                // Primary ray samples for view ray integration
    int lightSamples_ = 3;                  // Light ray samples for transmittance calculation
    float maxDistance_ = 400000.0f;         // Maximum ray marching distance
    
    // Atmospheric boundary layer (surface dust/aerosols)
    float boundaryLayerHeight_ = 800.0f;    // ABL height in meters
    float boundaryLayerDensityMultiplier_ = 0.2f; // ABL particle density multiplier
    float boundaryLayerFalloff_ = 1200.0f;  // ABL density falloff distance
    float dustExtinctionCoeff_ = 0.003f;    // Dust particle extinction coefficient
    float dustScatteringCoeff_ = 0.01f;     // Dust particle scattering coefficient
    float dustG_ = 0.9f;                    // Dust phase function asymmetry parameter

    // Multiple scattering approximation
    bool enableMultipleScattering_ = true;  // Enable/disable multiple scattering
    float multipleScatteringFactor_ = 0.15f; // Multiple scattering intensity
    float ambientScatteringFactor_ = 0.05f; // Ambient sky light contribution
    Vec3 skyAmbientColor_ = Vec3(0.4f, 0.6f, 1.0f); // Ambient sky color tint
    int multipleScatteringSamples_ = 4;     // Samples for multiple scattering integration

    AtmosphereContentSettings sunColorSettings_; // Sun color calculation settings
    
    // Water surface rendering
    float waterLevel_ = 0.0f;               // Sea level reference
    float waterRed_ = 0.05f;                // Water surface red component
    float waterGreen_ = 0.15f;              // Water surface green component
    float waterBlue_ = 0.35f;               // Water surface blue component
    
    // Rendering optimizations
    bool hemisphereOnly_ = false;           // Render only upper hemisphere for performance
};

/**
 * Pre-computed lookup tables for atmospheric calculations
 * Provides fast access to phase functions and density calculations
 */
class AtmosphereLUT {
private:
    static constexpr int PHASE_LUT_SIZE = 1024;   // Phase function lookup table size
    static constexpr int DENSITY_LUT_SIZE = 512;  // Density function lookup table size
    
    float rayleighPhaseLut_[PHASE_LUT_SIZE];      // Pre-computed Rayleigh phase values
    float miePhaseLut_[PHASE_LUT_SIZE];           // Pre-computed Mie phase values
    float densityLut_[DENSITY_LUT_SIZE];          // Pre-computed exponential density values
    
public:
    AtmosphereLUT(float mieG) {
        // Pre-compute Rayleigh and Mie phase functions
        for (int i = 0; i < PHASE_LUT_SIZE; ++i) {
            float cosTheta = (float(i) / (PHASE_LUT_SIZE - 1)) * 2.0f - 1.0f;
            
            // Rayleigh phase function: (3/16π)(1 + cos²θ)
            rayleighPhaseLut_[i] = (3.0f / (16.0f * M_PI)) * (1.0f + cosTheta * cosTheta);
            
            // Henyey-Greenstein phase function for Mie scattering
            float g2 = mieG * mieG;
            float denom = 1.0f + g2 - 2.0f * mieG * cosTheta;
            miePhaseLut_[i] = (1.0f - g2) / (4.0f * M_PI * std::pow(denom, 1.5f));
        }
        
        // Pre-compute exponential density falloff: exp(-t * 10)
        for (int i = 0; i < DENSITY_LUT_SIZE; ++i) {
            float t = float(i) / (DENSITY_LUT_SIZE - 1);
            densityLut_[i] = std::exp(-t * 10.0f);
        }
    }
    
    // Fast Rayleigh phase function lookup
    inline float
    rayleighPhase(float cosTheta) const {
        int idx = int((cosTheta + 1.0f) * 0.5f * (PHASE_LUT_SIZE - 1));
        idx = std::max(0, std::min(idx, PHASE_LUT_SIZE - 1));
        return rayleighPhaseLut_[idx];
    }
    
    // Fast Mie phase function lookup
    inline float
    miePhase(float cosTheta) const {
        int idx = int((cosTheta + 1.0f) * 0.5f * (PHASE_LUT_SIZE - 1));
        idx = std::max(0, std::min(idx, PHASE_LUT_SIZE - 1));
        return miePhaseLut_[idx];
    }
    
    // Fast exponential density lookup
    inline float
    expDensity(float scaleFactor) const {
        int idx = int(scaleFactor * (DENSITY_LUT_SIZE - 1) / 10.0f);
        idx = std::max(0, std::min(idx, DENSITY_LUT_SIZE - 1));
        return densityLut_[idx];
    }
};

/**
 * Perlin noise generator for volumetric cloud simulation
 * Generates pseudo-random noise patterns for natural-looking cloud formations
 */
class PerlinNoise {
private:
    std::vector<int> p_;    // Permutation table for noise generation
    
    // Linear interpolation
    inline float
    lerp(float a, float b, float t) const { 
        return a + t * (b - a); 
    }
    
    // Gradient function for noise calculation
    inline float
    grad(int hash, float x, float y) const {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : h == 12 || h == 14 ? x : 0;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    PerlinNoise(unsigned int seed = 12345) {
        // Initialize and shuffle permutation table
        p_.resize(256);
        for (int i = 0; i < 256; ++i) p_[i] = i;
        std::mt19937 rng(seed);
        std::shuffle(p_.begin(), p_.end(), rng);
        p_.insert(p_.end(), p_.begin(), p_.end());  // Duplicate for seamless wrapping
    }

    // Generate 2D Perlin noise value
    float
    noise(float x, float y) const {
        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        x -= std::floor(x);
        y -= std::floor(y);
        float u = x * x * (3.0f - 2.0f * x);  // Smoothstep interpolation
        float v = y * y * (3.0f - 2.0f * y);
        int A = p_[X] + Y, B = p_[X + 1] + Y;
        return lerp(
            lerp(grad(p_[A], x, y), grad(p_[B], x - 1, y), u),
            lerp(grad(p_[A + 1], x, y - 1), grad(p_[B + 1], x - 1, y - 1), u), v);
    }

    // Generate fractal Brownian motion (multiple octaves of noise)
    float
    fBm(float x, float y, int octaves = 4, float persistence = 0.5f, float lacunarity = 2.0f) const {
        float value = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float maxValue = 0.0f;
        
        for (int i = 0; i < octaves; ++i) {
            value += noise(x * frequency, y * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }
        return value / maxValue;
    }
};

// Global state variables
AtmosphereSettings gAtmosphere;  // Global atmosphere configuration
AtmosphereLUT* gLut = nullptr;   // Global lookup table instance

// Forward function declarations
Vec3 unifiedAtmosphericScatteringEnhanced(const Vec3& cameraPos, const Vec3& rayDir, 
                                          const Vec3& sunDir, const PerlinNoise& noise, 
                                          const AtmosphereSettings& settings);

Vec3 calculateMultipleScatteringWithSunColor(const Vec3& cameraPos, const Vec3& rayDir, 
                                             const Vec3& sunDir, const PerlinNoise& noise,
                                             const AtmosphereSettings& settings,
                                             const SunColorResult& sunColorInfo);

Vec3 calculateTransmittanceToSun(const Vec3& point, const Vec3& sunDir, 
                                const PerlinNoise& noise, 
                                const AtmosphereSettings& settings);

Vec3 addSunDisk(const Vec3& cameraPos, const Vec3& rayDir, const Vec3& sunDir, 
               const PerlinNoise& noise, const AtmosphereSettings& settings, const Vec3& skyColor);

/**
 * Convert screen coordinates to 3D ray direction
 * Maps 2D pixel coordinates to spherical coordinate system
 */
inline Vec3
screenToRay(int x, int y, int width, int height, bool hemisphereOnly) {
    float u = float(x) / width;   // Horizontal coordinate [0, 1]
    float v = float(y) / height;  // Vertical coordinate [0, 1]
    
    float phi = (u - 0.5f) * 2.0f * M_PI;                    // Azimuth angle
    float theta = v * M_PI * (hemisphereOnly ? 0.5f : 1.0f); // Elevation angle
    
    return Vec3(
        std::sin(theta) * std::cos(phi),  // X component
        std::cos(theta),                  // Y component (up)
        std::sin(theta) * std::sin(phi)   // Z component
    );
}

/**
 * Simple cloud phase function
 * Provides forward-scattering behavior for cloud particles
 */
inline float
cloudPhase(float cosTheta) {
    return 0.25f * (1.0f + 3.0f * std::max(0.0f, cosTheta));
}

/**
 * Sample atmospheric density at world position
 * Returns (Rayleigh density, Mie density, dust density) based on altitude
 */
Vec3
sampleAtmosphereDensity(const Vec3& worldPos, const AtmosphereSettings& settings) {
    float altitude = worldPos.length() - settings.planetRadius_;
    altitude = std::max(0.0f, altitude);  // Clamp to prevent negative altitudes
    
    // Exponential density falloff with altitude
    float rayleighDensity = gLut->expDensity(altitude / settings.rayleighScaleHeight_);
    float mieDensity = gLut->expDensity(altitude / settings.mieScaleHeight_);
    
    // Boundary layer dust with additional falloff above boundary layer height
    float dustDensity = gLut->expDensity(altitude / settings.boundaryLayerFalloff_) * 
                       settings.boundaryLayerDensityMultiplier_;
    
    if (altitude > settings.boundaryLayerHeight_) {
        float fadeFactor = gLut->expDensity((altitude - settings.boundaryLayerHeight_) / 1000.0f);
        dustDensity *= fadeFactor;
    }
    
    return Vec3(rayleighDensity, mieDensity, std::min(dustDensity, 0.3f));
}

/**
 * Sample cloud density using Perlin noise
 * Generates volumetric cloud formations with altitude-based distribution
 */
float
sampleCloudDensity(const Vec3& worldPos, const PerlinNoise& noise, const AtmosphereSettings& settings) {
    float altitude = worldPos.length() - settings.planetRadius_;
    
    // Early exit if outside cloud altitude range
    if (altitude < settings.cloudAltitudeMin_ || altitude > settings.cloudAltitudeMax_) {
        return 0.0f;
    }
    
    // Calculate altitude-based cloud density falloff
    float cloudMid = (settings.cloudAltitudeMin_ + settings.cloudAltitudeMax_) * 0.5f;
    float cloudRange = (settings.cloudAltitudeMax_ - settings.cloudAltitudeMin_) * 0.5f;
    float altitudeFactor = 1.0f - std::abs((altitude - cloudMid) / cloudRange);
    altitudeFactor = std::max(0.0f, altitudeFactor);
    
    // Generate fractal noise for cloud shapes
    float cloudNoise = noise.fBm(worldPos.x_ * settings.cloudScale_, 
                                worldPos.z_ * settings.cloudScale_, 3, 0.7f, 2.0f);
    
    // Apply threshold and shaping to noise
    float cloudDensity = std::max(0.0f, (cloudNoise - 0.3f) / 0.7f);
    cloudDensity = std::pow(cloudDensity, 2.0f) * altitudeFactor * settings.cloudDensityMultiplier_;
    
    return std::min(cloudDensity, 0.05f);  // Clamp maximum density
}

/**
 * Ray-sphere intersection test
 * Calculates intersection distances for ray with sphere
 */
inline bool
raySphereIntersection(const Vec3& rayOrigin, const Vec3& rayDir, float sphereRadius, 
                     float& tNear, float& tFar) {
    float a = rayDir.dot(rayDir);
    float b = 2.0f * rayOrigin.dot(rayDir);
    float c = rayOrigin.dot(rayOrigin) - sphereRadius * sphereRadius;
    
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return false;  // No intersection
    
    float sqrtDiscriminant = std::sqrt(discriminant);
    tNear = (-b - sqrtDiscriminant) / (2.0f * a);
    tFar = (-b + sqrtDiscriminant) / (2.0f * a);
    
    return true;
}

/**
 * Calculate light transmittance from point to sun
 * Accounts for atmospheric extinction along the light ray path
 */
Vec3
calculateTransmittanceToSun(const Vec3& point, const Vec3& sunDir, const PerlinNoise& noise, 
                           const AtmosphereSettings& settings) {
    float tNear, tFar;
    float atmosphereRadius = settings.planetRadius_ + settings.atmosphereHeight_;
    
    // Test intersection with atmosphere sphere
    if (!raySphereIntersection(point, sunDir, atmosphereRadius, tNear, tFar)) {
        return Vec3(1.0f, 1.0f, 1.0f);  // No atmosphere in light path
    }
    
    float rayLength = tFar - std::max(0.0f, tNear);
    float stepSize = rayLength / settings.lightSamples_;
    
    Vec3 opticalDepth(0.0f, 0.0f, 0.0f);
    
    // Cache coefficients for performance
    const float rayleighR = settings.rayleighCoeffR_;
    const float rayleighG = settings.rayleighCoeffG_;
    const float rayleighB = settings.rayleighCoeffB_;
    const float mieCoeff = settings.mieCoefficient_;
    const float dustExtinctCoeff = settings.dustExtinctionCoeff_;
    const float cloudExtinctCoeff = settings.cloudExtinctionCoeff_;
    
    // Ray march through atmosphere to accumulate optical depth
    for (int i = 0; i < settings.lightSamples_; ++i) {
        float t = std::max(0.0f, tNear) + (i + 0.5f) * stepSize;
        Vec3 samplePoint = point + sunDir * t;
        
        Vec3 atmoDensity = sampleAtmosphereDensity(samplePoint, settings);
        float cloudDensity = sampleCloudDensity(samplePoint, noise, settings);
        
        // Accumulate wavelength-dependent Rayleigh extinction
        float rayleighDensity = atmoDensity.x_;
        opticalDepth.x_ += rayleighDensity * rayleighR * stepSize;
        opticalDepth.y_ += rayleighDensity * rayleighG * stepSize;
        opticalDepth.z_ += rayleighDensity * rayleighB * stepSize;
        
        // Accumulate wavelength-independent extinction (Mie, dust, clouds)
        float totalExtinction = (atmoDensity.y_ * mieCoeff +
                                atmoDensity.z_ * dustExtinctCoeff +
                                cloudDensity * cloudExtinctCoeff) * stepSize;
        
        opticalDepth.x_ += totalExtinction;
        opticalDepth.y_ += totalExtinction;
        opticalDepth.z_ += totalExtinction;
    }
    
    // Apply Beer-Lambert law: T = exp(-τ)
    return Vec3(std::exp(-opticalDepth.x_), 
               std::exp(-opticalDepth.y_), 
               std::exp(-opticalDepth.z_));
}

/**
 * Calculate dynamic sun color based on atmospheric content
 * Analyzes atmospheric particles along sun ray to determine color shift and visibility
 */
SunColorResult
calculateSunColorFromAtmosphere(const Vec3& cameraPos, const Vec3& sunDir, 
                               const PerlinNoise& noise, const AtmosphereSettings& settings) {
    SunColorResult result;
    result.sunColorMultiplier_ = Vec3(1.0f, 1.0f, 1.0f);  // Default white sun
    result.sunVisibility_ = 1.0f;                          // Default fully visible
    result.totalContent_ = 0.0f;                           // Default no content
    
    float sunElevation = sunDir.y_;
    const auto& colorSettings = settings.sunColorSettings_;
    
    // Adaptive sampling: more samples for low sun angles (longer atmospheric path)
    int sunSamples;
    if (sunElevation < colorSettings.lowSunThreshold_) {
        sunSamples = colorSettings.maxSunSamples_;
    } else {
        float t = (sunElevation - colorSettings.lowSunThreshold_) / 
                 (1.0f - colorSettings.lowSunThreshold_);
        sunSamples = int(colorSettings.maxSunSamples_ * (1.0f - t) + 
                        colorSettings.minSunSamples_ * t);
    }
    
    // Calculate ray intersection with atmosphere
    float tNear, tFar;
    float atmosphereRadius = settings.planetRadius_ + settings.atmosphereHeight_;
    
    if (!raySphereIntersection(cameraPos, sunDir, atmosphereRadius, tNear, tFar)) {
        return result;  // No atmosphere between camera and sun
    }
    
    if (tNear < 0.0f) tNear = 0.0f;
    float rayLength = tFar - tNear;
    float stepSize = rayLength / sunSamples;
    
    float totalContent = 0.0f;
    
    // Ray march to accumulate atmospheric content
    for (int i = 0; i < sunSamples; ++i) {
        float t = tNear + (i + 0.5f) * stepSize;
        Vec3 samplePoint = cameraPos + sunDir * t;
        
        Vec3 atmoDensity = sampleAtmosphereDensity(samplePoint, settings);
        float cloudDensity = sampleCloudDensity(samplePoint, noise, settings);
        
        // Weight different particle types for color contribution
        float ablContribution = atmoDensity.z_ * colorSettings.ablReddeningFactor_;
        float cloudContribution = cloudDensity * colorSettings.cloudReddeningFactor_;
        
        totalContent += (ablContribution + cloudContribution) * stepSize;
    }
    
    result.totalContent_ = totalContent;
    
    // Calculate color temperature shift based on atmospheric content
    if (colorSettings.useExponentialMapping_) {
        // Exponential mapping for physically-based color shift
        result.sunColorMultiplier_.x_ = std::exp(-totalContent * colorSettings.redExtinctionRate_);
        result.sunColorMultiplier_.y_ = std::exp(-totalContent * colorSettings.greenExtinctionRate_);
        result.sunColorMultiplier_.z_ = std::exp(-totalContent * colorSettings.blueExtinctionRate_);
    } else {
        // Linear mapping with clamped minimums for stable results
        result.sunColorMultiplier_.x_ = std::max(0.65f, 1.0f - totalContent * colorSettings.redExtinctionRate_);
        result.sunColorMultiplier_.y_ = std::max(0.35f, 1.0f - totalContent * colorSettings.greenExtinctionRate_);
        result.sunColorMultiplier_.z_ = std::max(0.15f, 1.0f - totalContent * colorSettings.blueExtinctionRate_);
    }
    
    // Calculate sun visibility based on atmospheric content
    float visibilityContent = std::min(totalContent, 20.0f);  // Clamp for stability
    
    if (visibilityContent > colorSettings.totalBlockageThreshold_) {
        result.sunVisibility_ = 0.4f;  // Minimum visibility (heavily obscured)
    } else {
        result.sunVisibility_ = std::exp(-visibilityContent * colorSettings.visibilityFalloffRate_);
        result.sunVisibility_ = std::max(0.4f, std::min(1.0f, result.sunVisibility_));
    }
    
    return result;
}

/**
 * Add sun disk to sky color
 * Renders bright sun disk with atmospheric color filtering and angular falloff
 */
Vec3
addSunDisk(const Vec3& cameraPos, const Vec3& rayDir, const Vec3& sunDir, 
          const PerlinNoise& noise, const AtmosphereSettings& settings, const Vec3& skyColor) {
    
    float cosAngle = rayDir.dot(sunDir);
    float sunAngularRadius = 0.0045f;  // Sun's angular radius in radians (~0.25 degrees)
    
    // Check if ray direction intersects sun disk
    if (cosAngle > std::cos(sunAngularRadius)) {
        // Calculate sun appearance through atmosphere
        SunColorResult sunAppearance = calculateSunColorFromAtmosphere(cameraPos, sunDir, noise, settings);
        
        // Calculate effective sun intensity with atmospheric attenuation
        float effectiveSunIntensity = settings.sunIntensity_ * sunAppearance.sunVisibility_ * 50.0f;
        Vec3 sunDiskColor = Vec3(effectiveSunIntensity, effectiveSunIntensity, effectiveSunIntensity) * 
                           sunAppearance.sunColorMultiplier_;
        
        // Calculate smooth falloff from center to edge of sun disk
        float angleFromCenter = std::acos(cosAngle);
        float falloff = std::max(0.0f, (sunAngularRadius - angleFromCenter) / sunAngularRadius);
        falloff = std::pow(falloff, 0.5f);  // Smooth falloff curve
        
        return skyColor + sunDiskColor * falloff;
    }
    
    return skyColor;  // Ray doesn't intersect sun disk
}

/**
 * Main atmospheric scattering calculation
 * Unified function that handles Rayleigh, Mie, dust, and cloud scattering
 * with dynamic sun coloring and multiple scattering approximation
 */
Vec3
unifiedAtmosphericScatteringEnhanced(const Vec3& cameraPos, const Vec3& rayDir, 
                                     const Vec3& sunDir, const PerlinNoise& noise, 
                                     const AtmosphereSettings& settings) {
    
    float tNear, tFar;
    float atmosphereRadius = settings.planetRadius_ + settings.atmosphereHeight_;
    
    // Test ray intersection with atmosphere
    if (!raySphereIntersection(cameraPos, rayDir, atmosphereRadius, tNear, tFar)) {
        return Vec3(0.0f, 0.0f, 0.0f);  // Ray misses atmosphere
    }
    
    if (tNear < 0.0f) tNear = 0.0f;  // Handle camera inside atmosphere
    
    float rayLength = std::min(tFar - tNear, settings.maxDistance_);
    float stepSize = rayLength / settings.primarySamples_;
    
    Vec3 scatteredLight(0.0f, 0.0f, 0.0f);  // Accumulated scattered light
    Vec3 transmittance(1.0f, 1.0f, 1.0f);   // Light transmission factor
    
    float cosTheta = rayDir.dot(sunDir);  // Scattering angle
    
    // Pre-calculate phase function values for performance
    float rayleighPhaseVal = gLut->rayleighPhase(cosTheta);
    float miePhaseVal = gLut->miePhase(cosTheta);
    float cloudPhaseVal = cloudPhase(cosTheta);
    
    // Cache frequently used coefficients
    float baseIntensity = settings.sunIntensity_;
    const float rayleighR = settings.rayleighCoeffR_;
    const float rayleighG = settings.rayleighCoeffG_;
    const float rayleighB = settings.rayleighCoeffB_;
    const float mieCoeff = settings.mieCoefficient_;
    const float dustScatterCoeff = settings.dustScatteringCoeff_;
    const float dustExtinctCoeff = settings.dustExtinctionCoeff_;
    const float cloudScatterCoeff = settings.cloudScatteringCoeff_;
    const float cloudExtinctCoeff = settings.cloudExtinctionCoeff_;
    
    // Ray marching through atmosphere
    for (int i = 0; i < settings.primarySamples_; ++i) {
        float t = tNear + (i + 0.5f) * stepSize;
        Vec3 samplePoint = cameraPos + rayDir * t;
        
        // Sample atmospheric densities at current point
        Vec3 atmoDensity = sampleAtmosphereDensity(samplePoint, settings);
        float cloudDensity = sampleCloudDensity(samplePoint, noise, settings);
        
        // Extract individual density components
        float rayleighDensity = atmoDensity.x_;
        float mieDensity = atmoDensity.y_;
        float dustDensity = atmoDensity.z_;
        
        // Calculate scattering coefficients for each component
        Vec3 rayleighScattering(
            rayleighDensity * rayleighR,
            rayleighDensity * rayleighG, 
            rayleighDensity * rayleighB
        );
        
        float mieScattering = mieDensity * mieCoeff;
        float dustScattering = dustDensity * dustScatterCoeff;
        float cloudScattering = cloudDensity * cloudScatterCoeff;
        
        // Combine all scattering contributions with phase functions
        Vec3 totalScattering = rayleighScattering * rayleighPhaseVal + 
                              Vec3(mieScattering, mieScattering, mieScattering) * miePhaseVal +
                              Vec3(dustScattering, dustScattering, dustScattering) * miePhaseVal +
                              Vec3(cloudScattering, cloudScattering, cloudScattering) * cloudPhaseVal;
        
        // Calculate dynamic sun color at this sample point
        SunColorResult localSunColor = calculateSunColorFromAtmosphere(samplePoint, sunDir, noise, settings);
        
        // Apply sun color and atmospheric attenuation
        float effectiveSunIntensity = baseIntensity * localSunColor.sunVisibility_;
        Vec3 coloredSunLight = Vec3(effectiveSunIntensity, effectiveSunIntensity, effectiveSunIntensity) * 
                              localSunColor.sunColorMultiplier_;
        
        // Calculate in-scattered light contribution
        Vec3 inScattered = totalScattering * coloredSunLight * stepSize;
        
        // Clamp to prevent numerical overflow
        inScattered.x_ = std::min(inScattered.x_, 10.0f);
        inScattered.y_ = std::min(inScattered.y_, 10.0f);
        inScattered.z_ = std::min(inScattered.z_, 10.0f);
        
        // Accumulate scattered light with current transmittance
        scatteredLight.x_ += inScattered.x_ * transmittance.x_;
        scatteredLight.y_ += inScattered.y_ * transmittance.y_;
        scatteredLight.z_ += inScattered.z_ * transmittance.z_;
        
        // Calculate total extinction for transmittance update
        float totalExtinction = mieScattering + 
                               dustDensity * dustExtinctCoeff + 
                               cloudDensity * cloudExtinctCoeff;
        
        Vec3 extinction = rayleighScattering + Vec3(totalExtinction, totalExtinction, totalExtinction);
        
        // Update transmittance using Beer-Lambert law
        transmittance.x_ *= std::exp(-extinction.x_ * stepSize);
        transmittance.y_ *= std::exp(-extinction.y_ * stepSize);
        transmittance.z_ *= std::exp(-extinction.z_ * stepSize);
    }
    
    // Add multiple scattering approximation
    SunColorResult sunColorInfo = calculateSunColorFromAtmosphere(cameraPos, sunDir, noise, settings);
    Vec3 multipleScattered = calculateMultipleScatteringWithSunColor(
        cameraPos, rayDir, sunDir, noise, settings, sunColorInfo);
    
    return scatteredLight + multipleScattered;
}

/**
 * Calculate multiple scattering approximation
 * Simulates light that has bounced multiple times through the atmosphere
 * Includes both ambient sky light and second-order sun scattering
 */
Vec3
calculateMultipleScatteringWithSunColor(const Vec3& cameraPos, const Vec3& rayDir, 
                                       const Vec3& sunDir, const PerlinNoise& noise,
                                       const AtmosphereSettings& settings,
                                       const SunColorResult& sunColorInfo) {
    if (!settings.enableMultipleScattering_) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    
    float tNear, tFar;
    float atmosphereRadius = settings.planetRadius_ + settings.atmosphereHeight_;
    
    if (!raySphereIntersection(cameraPos, rayDir, atmosphereRadius, tNear, tFar)) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    
    if (tNear < 0.0f) tNear = 0.0f;
    float rayLength = std::min(tFar - tNear, settings.maxDistance_);
    
    int msSamples = settings.multipleScatteringSamples_;
    float stepSize = rayLength / msSamples;
    
    Vec3 multipleContribution(0.0f, 0.0f, 0.0f);
    Vec3 transmittance(1.0f, 1.0f, 1.0f);
    
    // Pre-calculate colored sun intensity
    float effectiveSunIntensity = settings.sunIntensity_ * sunColorInfo.sunVisibility_;
    Vec3 coloredSunIntensity = Vec3(effectiveSunIntensity, effectiveSunIntensity, effectiveSunIntensity) * 
                               sunColorInfo.sunColorMultiplier_;
    
    for (int i = 0; i < msSamples; ++i) {
        float t = tNear + (i + 0.5f) * stepSize;
        Vec3 samplePoint = cameraPos + rayDir * t;
        
        Vec3 atmoDensity = sampleAtmosphereDensity(samplePoint, settings);
        float cloudDensity = sampleCloudDensity(samplePoint, noise, settings);
        
        // Calculate average scattering coefficient
        float totalScattering = atmoDensity.x_ * (settings.rayleighCoeffR_ + settings.rayleighCoeffG_ + settings.rayleighCoeffB_) / 3.0f +
                               atmoDensity.y_ * settings.mieCoefficient_ +
                               atmoDensity.z_ * settings.dustScatteringCoeff_ +
                               cloudDensity * settings.cloudScatteringCoeff_;
        
        // Multiple scattering phase function (more isotropic than single scattering)
        float cosTheta = rayDir.dot(sunDir);
        float msPhase = 0.25f * (1.0f + 0.3f * cosTheta);
        
        // Ambient sky contribution (simulates omnidirectional scattered light)
        Vec3 coloredAmbient = settings.skyAmbientColor_ * sunColorInfo.sunColorMultiplier_;
        Vec3 ambientContribution = coloredAmbient * 
                                  settings.ambientScatteringFactor_ *
                                  totalScattering * stepSize;
        
        // Second-order sun scattering (light scattered once then again)
        float sunElevation = std::max(0.0f, sunDir.y_);
        Vec3 sunMsContribution = coloredSunIntensity * 0.1f *
                                settings.multipleScatteringFactor_ *
                                msPhase * totalScattering * stepSize * 
                                (0.5f + 0.5f * sunElevation);  // Stronger when sun is higher
        
        // Accumulate multiple scattering contributions
        multipleContribution.x_ += (ambientContribution.x_ + sunMsContribution.x_) * transmittance.x_;
        multipleContribution.y_ += (ambientContribution.y_ + sunMsContribution.y_) * transmittance.y_;
        multipleContribution.z_ += (ambientContribution.z_ + sunMsContribution.z_) * transmittance.z_;
        
        // Update transmittance
        Vec3 extinction(totalScattering * stepSize);
        transmittance.x_ *= std::exp(-extinction.x_);
        transmittance.y_ *= std::exp(-extinction.y_);
        transmittance.z_ *= std::exp(-extinction.z_);
    }
    
    return multipleContribution;
}

/**
 * Convert floating-point RGB to RGBE format
 * RGBE stores HDR colors using shared exponent encoding
 */
void
floatToRgbe(uint8_t* rgbe, float r, float g, float b) {
    float v = std::max(std::max(r, g), b);  // Find maximum component
    
    if (v < 1e-32f) {
        // Handle very small values
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
    } else {
        int e;
        v = std::frexp(v, &e) * 256.0f / v;  // Extract and normalize exponent
        
        // Encode RGB components with shared exponent
        rgbe[0] = static_cast<uint8_t>(std::min(255.0f, r * v));
        rgbe[1] = static_cast<uint8_t>(std::min(255.0f, g * v));
        rgbe[2] = static_cast<uint8_t>(std::min(255.0f, b * v));
        rgbe[3] = static_cast<uint8_t>(e + 128);  // Biased exponent
    }
}

/**
 * Configure atmosphere settings for vibrant sunset rendering
 * Sets up parameters optimized for bright, colorful sunset scenes
 */
void
setupBrightRedSunsetSettings(AtmosphereSettings& settings) {
    settings.boundaryLayerHeight_ = 800.0f;     // Moderate boundary layer height
    settings.cloudAltitudeMin_ = 1000.0f;       // Cloud layer minimum altitude
    settings.cloudAltitudeMax_ = 8000.0f;       // Cloud layer maximum altitude
}

/**
 * Generate HDRI image with atmospheric scattering
 * Main rendering function that produces HDR environment map
 */
void
generateHdriBrightRedSunset(int width, int height, const char* filename, const AtmosphereSettings& settings) {
    std::vector<uint8_t> image(width * height * 4);  // RGBE format (4 bytes per pixel)
    PerlinNoise noise;  // Noise generator for clouds

    // Initialize lookup tables
    if (gLut) delete gLut;
    gLut = new AtmosphereLUT(settings.mieG_);
    
    // Convert sun angles to direction vector
    float sunElevationRad = settings.sunElevationDegrees_ * M_PI / 180.0f;
    float sunAzimuthRad = settings.sunAzimuthDegrees_ * M_PI / 180.0f;
    
    Vec3 sunDir(
        std::cos(sunElevationRad) * std::cos(sunAzimuthRad),
        std::sin(sunElevationRad),
        std::cos(sunElevationRad) * std::sin(sunAzimuthRad)
    );
    sunDir = sunDir.normalize();
    
    // Position camera slightly above planet surface
    Vec3 cameraPos(0, settings.planetRadius_ + 10.0f, 0);
    
    // Calculate sun appearance for debugging output
    SunColorResult sunInfo = calculateSunColorFromAtmosphere(cameraPos, sunDir, noise, settings);
    
    // Print rendering configuration and sun analysis
    std::cout << "=== HDRI ATMOSPHERIC SCATTERING RENDERER ===\n";
    std::cout << "Rendering mode: " << (settings.hemisphereOnly_ ? "HEMISPHERE ONLY" : "FULL SPHERE") << "\n";
    std::cout << "Image size: " << width << "x" << height << " pixels\n";
    std::cout << "Sun elevation: " << settings.sunElevationDegrees_ << " degrees\n";
    std::cout << "Sun azimuth: " << settings.sunAzimuthDegrees_ << " degrees\n";
    std::cout << "Total atmospheric content: " << sunInfo.totalContent_ << "\n";
    std::cout << "Sun color multiplier: (" << sunInfo.sunColorMultiplier_.x_ << ", " 
              << sunInfo.sunColorMultiplier_.y_ << ", " << sunInfo.sunColorMultiplier_.z_ << ")\n";
    std::cout << "Sun visibility: " << sunInfo.sunVisibility_ << "\n";
    
    // Classify sun color based on blue component
    if (sunInfo.sunColorMultiplier_.z_ < 0.35f) {
        std::cout << "Sun color: DEEP RED (heavy atmospheric content)\n";
    } else if (sunInfo.sunColorMultiplier_.z_ < 0.55f) {
        std::cout << "Sun color: RED (moderate-heavy atmospheric content)\n"; 
    } else if (sunInfo.sunColorMultiplier_.z_ < 0.75f) {
        std::cout << "Sun color: ORANGE (moderate atmospheric content)\n";
    } else if (sunInfo.sunColorMultiplier_.z_ < 0.9f) {
        std::cout << "Sun color: YELLOW (light atmospheric content)\n";
    } else {
        std::cout << "Sun color: WHITE (clean atmosphere)\n";
    }

    int brightPixelCount = 0;    // Counter for bright pixels (quality metric)
    int skippedPixelCount = 0;   // Counter for optimization tracking
    constexpr int CHUNK_SIZE = 64;  // Process image in chunks for cache efficiency
    
    // Render image in cache-friendly chunks
    for (int chunkY = 0; chunkY < height; chunkY += CHUNK_SIZE) {
        int endY = std::min(chunkY + CHUNK_SIZE, height);
        for (int chunkX = 0; chunkX < width; chunkX += CHUNK_SIZE) {
            int endX = std::min(chunkX + CHUNK_SIZE, width);
            for (int y = chunkY; y < endY; ++y) {
                for (int x = chunkX; x < endX; ++x) {
                    float r = 0.0f, g = 0.0f, b = 0.0f;

                    // Convert pixel coordinates to ray direction
                    Vec3 rayDir = screenToRay(x, y, width, height, settings.hemisphereOnly_);
                    
                    // Check if ray points into sky (upward)
                    if (rayDir.y_ > 0.0f) {
                        // Render atmospheric scattering for sky pixels
                        #if ENABLE_UNIFIED_SCATTERING
                        Vec3 skyColor = unifiedAtmosphericScatteringEnhanced(
                            cameraPos, rayDir, sunDir, noise, settings);
                        
                        // Count bright pixels for quality assessment
                        if (skyColor.length() > 1.0f) {
                            brightPixelCount++;
                        }
                        
                        r = skyColor.x_;
                        g = skyColor.y_;
                        b = skyColor.z_;
                        #endif
                    } else {
                        // Handle ground/water pixels (ray points downward)
                        if (settings.hemisphereOnly_) {
                            // Hemisphere mode: simple ground color
                            r = g = b = 0.1f;
                            skippedPixelCount++;
                        } else {
                            // Full sphere mode: render water surface
                            #if ENABLE_WATER
                            float waterDepth = std::abs(rayDir.y_);  // Use ray angle for depth simulation
                            r = settings.waterRed_ + waterDepth * 0.1f;
                            g = settings.waterGreen_ + waterDepth * 0.1f;
                            b = settings.waterBlue_ + waterDepth * 0.1f;
                            #else
                            r = g = b = 0.1f;
                            #endif
                        }
                    }
                    
                    // Clamp values and convert to RGBE format
                    r = std::max(0.0f, r);
                    g = std::max(0.0f, g);
                    b = std::max(0.0f, b);
                    floatToRgbe(&image[(y * width + x) * 4], r, g, b);
                }
            }
        }
    }

    // Print rendering statistics
    std::cout << "Total bright pixels: " << brightPixelCount << "\n";
    std::cout << "Skipped ground pixels: " << skippedPixelCount;
    if (settings.hemisphereOnly_) {
        std::cout << " (performance gain: ~" << int(100.0f * skippedPixelCount / (width * height)) << "%)";
    }
    std::cout << "\n";

    // Write RGBE HDR file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open " << filename << " for writing\n";
        return;
    }
    
    // Write RGBE header
    file << "#?RADIANCE\n";
    file << "FORMAT=32-bit_rle_rgbe\n";
    file << "\n";
    file << "-Y " << height << " +X " << width << "\n";
    
    // Write image data
    file.write(const_cast<char*>(reinterpret_cast<const char*>(image.data())), image.size());
    file.close();
    
    std::cout << "HDRI generated successfully: " << filename << "\n\n";
    
    // Cleanup
    delete gLut;
    gLut = nullptr;
}

// Image dimensions for testing
#define SZ_X 512
#define SZ_Y 256

/**
 * Main function - demonstrates atmospheric scattering with various configurations
 * Generates test images showing different atmospheric conditions and sun positions
 */
int
main() {
    std::cout << "\n=== HDRI ATMOSPHERIC SCATTERING RENDERER ===\n";
    std::cout << "Generating test images with different atmospheric conditions\n\n";
    
    setupBrightRedSunsetSettings(gAtmosphere);
    
    // Test 1: Clean atmosphere, high sun - produces white/yellow sun
    std::cout << "Test 1: Clean atmosphere, high sun angle...\n";
    gAtmosphere.sunElevationDegrees_ = 35.0f;
    gAtmosphere.boundaryLayerDensityMultiplier_ = 0.05f;
    gAtmosphere.cloudDensityMultiplier_ = 0.02f;
    generateHdriBrightRedSunset(SZ_X, SZ_Y, "atmosphere_clean.hdr", gAtmosphere);
    
    // Test 2: Light atmosphere, moderate sun - produces yellow/orange sun
    std::cout << "Test 2: Light atmosphere, moderate sun angle...\n";
    gAtmosphere.sunElevationDegrees_ = 16.0f;
    gAtmosphere.boundaryLayerDensityMultiplier_ = 0.15f;
    gAtmosphere.cloudDensityMultiplier_ = 0.03f;
    generateHdriBrightRedSunset(SZ_X, SZ_Y, "atmosphere_light.hdr", gAtmosphere);
    
    // Test 3: Moderate atmosphere, low sun - produces orange sun
    std::cout << "Test 3: Moderate atmosphere, low sun angle...\n";
    gAtmosphere.sunElevationDegrees_ = 8.0f;
    gAtmosphere.boundaryLayerDensityMultiplier_ = 0.3f;
    gAtmosphere.cloudDensityMultiplier_ = 0.06f;
    generateHdriBrightRedSunset(SZ_X, SZ_Y, "atmosphere_moderate.hdr", gAtmosphere);
    
    // Test 4: Heavy atmosphere, low sun - produces red sun
    std::cout << "Test 4: Heavy atmosphere, low sun angle...\n";
    gAtmosphere.sunElevationDegrees_ = 4.0f;
    gAtmosphere.boundaryLayerDensityMultiplier_ = 0.5f;
    gAtmosphere.cloudDensityMultiplier_ = 0.12f;
    generateHdriBrightRedSunset(SZ_X, SZ_Y, "atmosphere_heavy.hdr", gAtmosphere);
    
    // Test 5: Very heavy atmosphere, very low sun - produces deep red sun
    std::cout << "Test 5: Very heavy atmosphere, horizon sun...\n";
    gAtmosphere.sunElevationDegrees_ = 1.0f;
    gAtmosphere.boundaryLayerDensityMultiplier_ = 0.8f;
    gAtmosphere.cloudDensityMultiplier_ = 0.25f;
    generateHdriBrightRedSunset(SZ_X, SZ_Y, "atmosphere_very_heavy.hdr", gAtmosphere);
    
    std::cout << "=== RENDERING COMPLETE ===\n";
    
    return 0;
}
