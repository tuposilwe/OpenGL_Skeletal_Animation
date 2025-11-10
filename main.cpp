#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in ivec4 aBoneIDs;
layout (location = 4) in vec4 aWeights;
layout (location = 5) in vec3 aTangent;
layout (location = 6) in vec3 aBitangent;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;
out mat3 TBN;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
const int MAX_BONES = 100;
uniform mat4 finalBonesMatrices[MAX_BONES];

void main()
{
    vec4 totalPosition = vec4(0.0);
    
    if(aBoneIDs[0] == -1) {
        totalPosition = vec4(aPos, 1.0);
    } else {
        for(int i = 0; i < 4; i++)
        {
            if(aBoneIDs[i] == -1) continue;
            if(aBoneIDs[i] >= MAX_BONES) {
                totalPosition = vec4(aPos, 1.0);
                break;
            }
            vec4 localPosition = finalBonesMatrices[aBoneIDs[i]] * vec4(aPos, 1.0);
            totalPosition += localPosition * aWeights[i];
        }
    }
    
    gl_Position = projection * view * model * totalPosition;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords;
    
    // Normal mapping
    vec3 T = normalize(mat3(model) * aTangent);
    vec3 B = normalize(mat3(model) * aBitangent);
    vec3 N = normalize(mat3(model) * aNormal);
    TBN = mat3(T, B, N);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
in mat3 TBN;

// Texture samplers
uniform sampler2D diffuseTexture;
uniform sampler2D normalTexture;
uniform sampler2D specularTexture;

// Lighting uniforms
uniform vec3 lightPos = vec3(0.0, 5.0, 0.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 viewPos;

// Material properties
uniform bool useDiffuseTexture = true;
uniform bool useNormalTexture = true;
uniform bool useSpecularTexture = true;
uniform vec3 objectColor = vec3(0.7, 0.5, 0.3);
uniform float shininess = 32.0;

void main()
{
    // Diffuse color
    vec4 diffuseColor;
    if(useDiffuseTexture) {
        diffuseColor = texture(diffuseTexture, TexCoords);
        if(diffuseColor.a < 0.1) 
            discard;
    } else {
        diffuseColor = vec4(objectColor, 1.0);
    }
    
    // Normal mapping
    vec3 normal;
    if(useNormalTexture) {
        // Get normal from normal map in [0,1] range
        normal = texture(normalTexture, TexCoords).rgb;
        // Transform normal vector to range [-1,1]
        normal = normalize(normal * 2.0 - 1.0);
        // Transform normal from tangent space to world space
        normal = normalize(TBN * normal);
    } else {
        normal = normalize(Normal);
    }
    
    // Specular map
    float specularStrength;
    if(useSpecularTexture) {
        specularStrength = texture(specularTexture, TexCoords).r;
    } else {
        specularStrength = 0.5;
    }
    
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Combine results
    vec3 result = (ambient + diffuse + specular) * diffuseColor.rgb;
    FragColor = vec4(result, diffuseColor.a);
}
)";

// Simple shader for ground
const char* groundVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = TexCoords;
}
)";

const char* groundFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

uniform sampler2D diffuseTexture;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // Ground texture with grid pattern
    vec2 uv = TexCoords * 10.0; // Scale texture
    vec4 texColor = texture(diffuseTexture, uv);
    
    // Add grid pattern
    vec2 grid = abs(fract(uv - 0.5) - 0.5);
    float line = min(grid.x, grid.y);
    float gridLine = 1.0 - smoothstep(0.0, 0.1, line);
    
    // Mix texture with grid
    vec3 color = mix(texColor.rgb, vec3(0.3, 0.3, 0.3), gridLine * 0.3);
    
    // Simple lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 ambient = 0.4 * lightColor;
    vec3 result = (ambient + diffuse) * color;
    
    FragColor = vec4(result, 1.0);
}
)";

struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
    std::string name;
};

struct BoneInfo {
    int id;
    glm::mat4 offset;
    glm::mat4 finalTransformation;
};

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::ivec4 BoneIDs = glm::ivec4(-1);
    glm::vec4 Weights = glm::vec4(0.0f);
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

struct KeyPosition {
    glm::vec3 position;
    float timeStamp;
};

struct KeyRotation {
    glm::quat orientation;
    float timeStamp;
};

struct KeyScale {
    glm::vec3 scale;
    float timeStamp;
};

class Bone {
private:
    std::vector<KeyPosition> m_Positions;
    std::vector<KeyRotation> m_Rotations;
    std::vector<KeyScale> m_Scales;
    int m_NumPositions;
    int m_NumRotations;
    int m_NumScalings;

    glm::mat4 m_LocalTransform;
    std::string m_Name;
    int m_ID;

public:
    Bone(const std::string& name, int ID, const aiNodeAnim* channel) : m_Name(name), m_ID(ID), m_LocalTransform(1.0f) {
        m_NumPositions = channel->mNumPositionKeys;
        for (int positionIndex = 0; positionIndex < m_NumPositions; ++positionIndex) {
            aiVector3D aiPosition = channel->mPositionKeys[positionIndex].mValue;
            float timeStamp = channel->mPositionKeys[positionIndex].mTime;
            KeyPosition data;
            data.position = glm::vec3(aiPosition.x, aiPosition.y, aiPosition.z);
            data.timeStamp = timeStamp;
            m_Positions.push_back(data);
        }

        m_NumRotations = channel->mNumRotationKeys;
        for (int rotationIndex = 0; rotationIndex < m_NumRotations; ++rotationIndex) {
            aiQuaternion aiOrientation = channel->mRotationKeys[rotationIndex].mValue;
            float timeStamp = channel->mRotationKeys[rotationIndex].mTime;
            KeyRotation data;
            data.orientation = glm::quat(aiOrientation.w, aiOrientation.x, aiOrientation.y, aiOrientation.z);
            data.timeStamp = timeStamp;
            m_Rotations.push_back(data);
        }

        m_NumScalings = channel->mNumScalingKeys;
        for (int keyIndex = 0; keyIndex < m_NumScalings; ++keyIndex) {
            aiVector3D scale = channel->mScalingKeys[keyIndex].mValue;
            float timeStamp = channel->mScalingKeys[keyIndex].mTime;
            KeyScale data;
            data.scale = glm::vec3(scale.x, scale.y, scale.z);
            data.timeStamp = timeStamp;
            m_Scales.push_back(data);
        }
    }

    void Update(float animationTime) {
        glm::mat4 translation = InterpolatePosition(animationTime);
        glm::mat4 rotation = InterpolateRotation(animationTime);
        glm::mat4 scale = InterpolateScaling(animationTime);
        m_LocalTransform = translation * rotation * scale;
    }

    glm::mat4 GetLocalTransform() { return m_LocalTransform; }
    std::string GetBoneName() const { return m_Name; }
    int GetBoneID() { return m_ID; }

    int GetPositionIndex(float animationTime) {
        for (int index = 0; index < m_NumPositions - 1; ++index) {
            if (animationTime < m_Positions[index + 1].timeStamp)
                return index;
        }
        return 0;
    }

    int GetRotationIndex(float animationTime) {
        for (int index = 0; index < m_NumRotations - 1; ++index) {
            if (animationTime < m_Rotations[index + 1].timeStamp)
                return index;
        }
        return 0;
    }

    int GetScaleIndex(float animationTime) {
        for (int index = 0; index < m_NumScalings - 1; ++index) {
            if (animationTime < m_Scales[index + 1].timeStamp)
                return index;
        }
        return 0;
    }

private:
    float GetScaleFactor(float lastTimeStamp, float nextTimeStamp, float animationTime) {
        float scaleFactor = 0.0f;
        float midWayLength = animationTime - lastTimeStamp;
        float framesDiff = nextTimeStamp - lastTimeStamp;
        scaleFactor = midWayLength / framesDiff;
        return scaleFactor;
    }

    glm::mat4 InterpolatePosition(float animationTime) {
        if (m_NumPositions == 1)
            return glm::translate(glm::mat4(1.0f), m_Positions[0].position);

        int p0Index = GetPositionIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = GetScaleFactor(m_Positions[p0Index].timeStamp,
            m_Positions[p1Index].timeStamp, animationTime);
        glm::vec3 finalPosition = glm::mix(m_Positions[p0Index].position,
            m_Positions[p1Index].position, scaleFactor);
        return glm::translate(glm::mat4(1.0f), finalPosition);
    }

    glm::mat4 InterpolateRotation(float animationTime) {
        if (m_NumRotations == 1) {
            auto rotation = glm::normalize(m_Rotations[0].orientation);
            return glm::mat4(rotation);
        }

        int p0Index = GetRotationIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = GetScaleFactor(m_Rotations[p0Index].timeStamp,
            m_Rotations[p1Index].timeStamp, animationTime);
        glm::quat finalRotation = glm::slerp(m_Rotations[p0Index].orientation,
            m_Rotations[p1Index].orientation, scaleFactor);
        finalRotation = glm::normalize(finalRotation);
        return glm::mat4(finalRotation);
    }

    glm::mat4 InterpolateScaling(float animationTime) {
        if (m_NumScalings == 1)
            return glm::scale(glm::mat4(1.0f), m_Scales[0].scale);

        int p0Index = GetScaleIndex(animationTime);
        int p1Index = p0Index + 1;
        float scaleFactor = GetScaleFactor(m_Scales[p0Index].timeStamp,
            m_Scales[p1Index].timeStamp, animationTime);
        glm::vec3 finalScale = glm::mix(m_Scales[p0Index].scale, m_Scales[p1Index].scale, scaleFactor);
        return glm::scale(glm::mat4(1.0f), finalScale);
    }
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;
    unsigned int VAO, VBO, EBO;

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices,
        std::vector<Texture> textures = {}) {
        this->vertices = vertices;
        this->indices = indices;
        this->textures = textures;
        setupMesh();
    }

    void Draw(unsigned int shaderProgram) {
        // Bind textures
        unsigned int diffuseNr = 1;
        unsigned int normalNr = 1;
        unsigned int specularNr = 1;

        bool hasDiffuseTexture = false;
        bool hasNormalTexture = false;
        bool hasSpecularTexture = false;

        for (unsigned int i = 0; i < textures.size(); i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            std::string number;
            std::string name = textures[i].type;

            if (name == "texture_diffuse") {
                number = std::to_string(diffuseNr++);
                hasDiffuseTexture = true;
                glUniform1i(glGetUniformLocation(shaderProgram, "diffuseTexture"), i);
            }
            else if (name == "texture_normal") {
                number = std::to_string(normalNr++);
                hasNormalTexture = true;
                glUniform1i(glGetUniformLocation(shaderProgram, "normalTexture"), i);
            }
            else if (name == "texture_specular") {
                number = std::to_string(specularNr++);
                hasSpecularTexture = true;
                glUniform1i(glGetUniformLocation(shaderProgram, "specularTexture"), i);
            }

            glUniform1i(glGetUniformLocation(shaderProgram, (name + number).c_str()), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }

        // Set texture usage uniforms
        glUniform1i(glGetUniformLocation(shaderProgram, "useDiffuseTexture"), hasDiffuseTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "useNormalTexture"), hasNormalTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "useSpecularTexture"), hasSpecularTexture);

        // Draw mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glActiveTexture(GL_TEXTURE0);
    }

private:
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // Vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // Vertex bone IDs
        glEnableVertexAttribArray(3);
        glVertexAttribIPointer(3, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, BoneIDs));
        // Vertex weights
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Weights));
        // Vertex tangent
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // Vertex bitangent
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

        glBindVertexArray(0);
    }
};

class Ground {
public:
    unsigned int VAO, VBO, EBO;
    unsigned int textureID;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Ground() {
        createGround();
        createGroundTexture();
    }

    void Draw(unsigned int shaderProgram) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "diffuseTexture"), 0);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:
    void createGround() {
        float size = 50.0f;
        float y = 0.0f; // Ground level

        vertices = {
            // Positions          // Normals           // Texture Coords
            {glm::vec3(-size, y, -size), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
            {glm::vec3(size, y, -size),  glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(1.0f, 0.0f)},
            {glm::vec3(size, y, size),   glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(1.0f, 1.0f)},
            {glm::vec3(-size, y, size), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 1.0f)}
        };

        indices = {
            0, 1, 2,
            2, 3, 0
        };

        // Setup buffers
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // Vertex texture coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }

    void createGroundTexture() {
        const int texSize = 256;
        std::vector<unsigned char> data(texSize * texSize * 3);

        for (int y = 0; y < texSize; y++) {
            for (int x = 0; x < texSize; x++) {
                int index = (y * texSize + x) * 3;

                // Create a grass-like texture with some variation
                float r = 0.4f + (sin(x * 0.1f) * 0.1f);
                float g = 0.6f + (cos(y * 0.1f) * 0.1f);
                float b = 0.3f + (sin((x + y) * 0.05f) * 0.05f);

                // Add some dark spots for variation
                float noise = (sin(x * 0.3f) * cos(y * 0.3f)) * 0.1f;
                r += noise;
                g += noise;
                b += noise * 0.5f;

                data[index] = static_cast<unsigned char>(r * 255);
                data[index + 1] = static_cast<unsigned char>(g * 255);
                data[index + 2] = static_cast<unsigned char>(b * 255);
            }
        }

        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSize, texSize, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
};

class Animation {
public:
    float m_Duration;
    float m_TicksPerSecond;
    std::vector<Bone> m_Bones;
    std::map<std::string, BoneInfo> m_BoneInfoMap;
    std::string m_Name;

    Animation(const aiScene* scene, int animationIndex, const std::string& animationName)
        : m_Name(animationName) {
        aiAnimation* animation = scene->mAnimations[animationIndex];
        m_Duration = animation->mDuration;
        m_TicksPerSecond = animation->mTicksPerSecond != 0 ? animation->mTicksPerSecond : 25.0f;

        std::cout << "Loading animation " << animationIndex << ": " << animationName << std::endl;
        std::cout << "Duration: " << m_Duration << " ticks" << std::endl;
        std::cout << "Ticks per second: " << m_TicksPerSecond << std::endl;
        std::cout << "Number of channels: " << animation->mNumChannels << std::endl;

        ReadMissingBones(animation);
    }

    Bone* FindBone(const std::string& name) {
        for (auto& bone : m_Bones) {
            if (bone.GetBoneName() == name)
                return &bone;
        }
        return nullptr;
    }

    const std::string& GetName() const { return m_Name; }

private:
    void ReadMissingBones(const aiAnimation* animation) {
        int boneCount = 0;

        for (int i = 0; i < animation->mNumChannels; i++) {
            aiNodeAnim* channel = animation->mChannels[i];
            std::string boneName = channel->mNodeName.C_Str();

            if (m_BoneInfoMap.find(boneName) == m_BoneInfoMap.end()) {
                BoneInfo boneInfo;
                boneInfo.id = boneCount;
                boneInfo.offset = glm::mat4(1.0f);
                m_BoneInfoMap[boneName] = boneInfo;
                boneCount++;
            }

            m_Bones.push_back(Bone(channel->mNodeName.C_Str(), m_BoneInfoMap[channel->mNodeName.C_Str()].id, channel));
        }

        std::cout << "Loaded " << m_Bones.size() << " animated bones" << std::endl;
    }
};

class Model {
public:
    std::vector<Mesh> meshes;
    std::map<std::string, BoneInfo> boneInfoMap;
    std::vector<glm::mat4> boneMatrices;
    int boneCounter = 0;
    glm::vec3 modelSize = glm::vec3(1.0f);
    glm::vec3 modelCenter = glm::vec3(0.0f);
    std::vector<std::unique_ptr<Animation>> animations;
    std::string directory;
    std::map<std::string, Texture> embeddedTextures;

    Model(const std::string& path) {
        loadModel(path);
        boneMatrices.resize(100, glm::mat4(1.0f));
        calculateModelDimensions();
    }

    void Draw(unsigned int shaderProgram) {
        for (auto& mesh : meshes) {
            mesh.Draw(shaderProgram);
        }
    }

    float getScaleFactor() {
        float maxDimension = std::max(modelSize.x, std::max(modelSize.y, modelSize.z));
        return 2.0f / maxDimension;
    }

    glm::vec3 getCenter() {
        return modelCenter;
    }

    bool HasAnimations() const {
        return !animations.empty();
    }

    int GetAnimationCount() const {
        return animations.size();
    }

    const std::string& GetAnimationName(int index) const {
        if (index >= 0 && index < animations.size()) {
            return animations[index]->GetName();
        }
        static std::string empty = "";
        return empty;
    }

    const aiScene* m_Scene;

private:
    Assimp::Importer m_Importer;
    std::vector<Texture> textures_loaded;

    void loadModel(const std::string& path) {
        size_t lastSlash = path.find_last_of("/\\");
        directory = (lastSlash == std::string::npos) ? "" : path.substr(0, lastSlash);

        m_Scene = m_Importer.ReadFile(path,
            aiProcess_Triangulate |
            aiProcess_FlipUVs |
            aiProcess_CalcTangentSpace |
            aiProcess_GenSmoothNormals);

        if (!m_Scene || m_Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !m_Scene->mRootNode) {
            std::cout << "ERROR::ASSIMP:: " << m_Importer.GetErrorString() << std::endl;
            return;
        }

        // Load all embedded textures first
        loadEmbeddedTextures();

        // Load all animations
        if (m_Scene->HasAnimations()) {
            std::cout << "Model has " << m_Scene->mNumAnimations << " animations" << std::endl;

            for (unsigned int i = 0; i < m_Scene->mNumAnimations; i++) {
                aiAnimation* aiAnim = m_Scene->mAnimations[i];
                std::string animName = aiAnim->mName.C_Str();

                // Handle NLA strip names - they often have prefixes
                if (animName.empty()) {
                    animName = "Animation_" + std::to_string(i);
                }
                else {
                    // Clean up NLA strip names (remove track prefixes if needed)
                    size_t lastDot = animName.find_last_of('.');
                    if (lastDot != std::string::npos) {
                        animName = animName.substr(lastDot + 1);
                    }
                }

                std::cout << "Loading animation " << i << ": " << animName
                    << " (Duration: " << aiAnim->mDuration << " ticks)" << std::endl;

                animations.push_back(std::make_unique<Animation>(m_Scene, i, animName));
            }
        }
        else {
            std::cout << "Model has no animations" << std::endl;
        }

        processNode(m_Scene->mRootNode, m_Scene);
    }

    void loadEmbeddedTextures() {
        if (m_Scene->HasTextures()) {
            std::cout << "Loading " << m_Scene->mNumTextures << " embedded textures..." << std::endl;
            for (unsigned int i = 0; i < m_Scene->mNumTextures; i++) {
                const aiTexture* texture = m_Scene->mTextures[i];
                if (texture) {
                    std::string textureName = texture->mFilename.C_Str();
                    std::cout << "Loading embedded texture: " << textureName << std::endl;

                    Texture embeddedTexture;
                    embeddedTexture.id = TextureFromEmbedded(texture);

                    // Determine texture type based on filename
                    if (textureName.find("normal") != std::string::npos ||
                        textureName.find("Normal") != std::string::npos) {
                        embeddedTexture.type = "texture_normal";
                    }
                    else if (textureName.find("specular") != std::string::npos ||
                        textureName.find("spec") != std::string::npos ||
                        textureName.find("Specular") != std::string::npos) {
                        embeddedTexture.type = "texture_specular";
                    }
                    else {
                        embeddedTexture.type = "texture_diffuse";
                    }

                    embeddedTexture.path = textureName;
                    embeddedTexture.name = getTextureName(textureName);

                    embeddedTextures[embeddedTexture.name] = embeddedTexture;
                    textures_loaded.push_back(embeddedTexture);

                    std::cout << "Successfully loaded embedded texture: " << textureName
                        << " as type: " << embeddedTexture.type << std::endl;
                }
            }
        }
    }

    std::string getTextureName(const std::string& path) {
        size_t lastSlash = path.find_last_of("/\\");
        size_t lastDot = path.find_last_of(".");
        std::string name = path.substr(lastSlash + 1, lastDot - lastSlash - 1);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        return name;
    }

    void processNode(aiNode* node, const aiScene* scene) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene);
        }
    }

    Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        std::vector<Texture> textures;

        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);

            if (mesh->HasNormals()) {
                vertex.Normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            }

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            // Extract tangents and bitangents for normal mapping
            if (mesh->HasTangentsAndBitangents()) {
                vertex.Tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
                vertex.Bitangent = glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);
            }
            else {
                vertex.Tangent = glm::vec3(1.0f, 0.0f, 0.0f);
                vertex.Bitangent = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        // Process materials and textures
        if (mesh->mMaterialIndex >= 0) {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

            // Load diffuse textures
            std::vector<Texture> diffuseMaps = loadMaterialTextures(material,
                aiTextureType_DIFFUSE, "texture_diffuse");
            textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

            // Load normal maps
            std::vector<Texture> normalMaps = loadMaterialTextures(material,
                aiTextureType_NORMALS, "texture_normal");
            textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

            // Load height maps (often used as normal maps)
            std::vector<Texture> heightMaps = loadMaterialTextures(material,
                aiTextureType_HEIGHT, "texture_normal");
            textures.insert(textures.end(), heightMaps.begin(), heightMaps.end());

            // Load specular maps
            std::vector<Texture> specularMaps = loadMaterialTextures(material,
                aiTextureType_SPECULAR, "texture_specular");
            textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
        }

        if (mesh->HasBones()) {
            std::cout << "Processing " << mesh->mNumBones << " bones for mesh" << std::endl;
            for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                aiBone* bone = mesh->mBones[i];
                std::string boneName = bone->mName.C_Str();

                if (boneInfoMap.find(boneName) == boneInfoMap.end()) {
                    BoneInfo boneInfo;
                    boneInfo.id = boneCounter;
                    boneInfo.offset = aiMatrixToGlm(bone->mOffsetMatrix);
                    boneInfoMap[boneName] = boneInfo;
                    boneCounter++;
                }

                for (unsigned int j = 0; j < bone->mNumWeights; j++) {
                    aiVertexWeight weight = bone->mWeights[j];
                    unsigned int vertexId = weight.mVertexId;
                    float boneWeight = weight.mWeight;

                    if (vertexId < vertices.size()) {
                        for (int k = 0; k < 4; k++) {
                            if (vertices[vertexId].BoneIDs[k] == -1) {
                                vertices[vertexId].BoneIDs[k] = boneInfoMap[boneName].id;
                                vertices[vertexId].Weights[k] = boneWeight;
                                break;
                            }
                        }
                    }
                }
            }
        }

        return Mesh(vertices, indices, textures);
    }

    std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName) {
        std::vector<Texture> textures;

        unsigned int textureCount = mat->GetTextureCount(type);
        std::cout << "Material has " << textureCount << " textures of type " << typeName << std::endl;

        for (unsigned int i = 0; i < textureCount; i++) {
            aiString str;
            mat->GetTexture(type, i, &str);
            std::string texturePath = str.C_Str();
            std::string textureName = getTextureName(texturePath);

            std::cout << "Looking for texture: " << texturePath << " (name: " << textureName << ")" << std::endl;

            // First try to find in embedded textures
            if (embeddedTextures.find(textureName) != embeddedTextures.end()) {
                std::cout << "Found matching embedded texture: " << textureName << std::endl;
                textures.push_back(embeddedTextures[textureName]);
                continue;
            }

            // Check if already loaded as external texture
            bool skip = false;
            for (unsigned int j = 0; j < textures_loaded.size(); j++) {
                if (std::strcmp(textures_loaded[j].path.data(), texturePath.c_str()) == 0) {
                    textures.push_back(textures_loaded[j]);
                    skip = true;
                    std::cout << "Texture already loaded, reusing: " << texturePath << std::endl;
                    break;
                }
            }

            if (!skip) {
                // Try to load as external texture
                Texture texture;
                texture.id = TextureFromFile(texturePath, directory);
                texture.type = typeName;
                texture.path = texturePath;
                texture.name = textureName;
                textures.push_back(texture);
                textures_loaded.push_back(texture);
            }
        }

        return textures;
    }

    unsigned int TextureFromEmbedded(const aiTexture* embeddedTexture) {
        unsigned int textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        if (embeddedTexture->mHeight == 0) {
            int width, height, nrComponents;
            unsigned char* data = stbi_load_from_memory(
                reinterpret_cast<unsigned char*>(embeddedTexture->pcData),
                embeddedTexture->mWidth,
                &width, &height, &nrComponents, 0);

            if (data) {
                GLenum format;
                if (nrComponents == 1) format = GL_RED;
                else if (nrComponents == 3) format = GL_RGB;
                else if (nrComponents == 4) format = GL_RGBA;

                glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
                glGenerateMipmap(GL_TEXTURE_2D);
                stbi_image_free(data);

                std::cout << "Embedded texture dimensions: " << width << "x" << height << ", components: " << nrComponents << std::endl;
            }
            else {
                std::cout << "Failed to load embedded texture using stbi_load_from_memory" << std::endl;
                createDefaultTexture(textureID);
            }
        }
        else {
            std::cout << "Uncompressed embedded texture format not supported" << std::endl;
            createDefaultTexture(textureID);
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return textureID;
    }

    unsigned int TextureFromFile(const std::string& filename, const std::string& directory) {
        std::vector<std::string> possiblePaths = {
            directory + '/' + filename,
            filename,
            directory + "/../textures/" + filename,
            "textures/" + filename,
            "../textures/" + filename
        };

        unsigned int textureID;
        glGenTextures(1, &textureID);

        int width, height, nrComponents;
        unsigned char* data = nullptr;
        std::string successfulPath;

        for (const auto& path : possiblePaths) {
            stbi_set_flip_vertically_on_load(true);
            data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
            if (data) {
                successfulPath = path;
                break;
            }
        }

        if (data) {
            GLenum format;
            if (nrComponents == 1) format = GL_RED;
            else if (nrComponents == 3) format = GL_RGB;
            else if (nrComponents == 4) format = GL_RGBA;

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            stbi_image_free(data);
            std::cout << "Loaded external texture: " << successfulPath
                << " (" << width << "x" << height << ")" << std::endl;
        }
        else {
            std::cout << "Texture failed to load at all attempted paths for: " << filename << std::endl;
            createDefaultTexture(textureID);
        }

        return textureID;
    }

    void createDefaultTexture(unsigned int textureID) {
        const int texSize = 64;
        std::vector<unsigned char> defaultData(texSize * texSize * 3);

        for (int y = 0; y < texSize; y++) {
            for (int x = 0; x < texSize; x++) {
                int index = (y * texSize + x) * 3;

                float r = (sin(x * 0.3f) * 0.5f + 0.5f) * 255;
                float g = (cos(y * 0.3f) * 0.5f + 0.5f) * 255;
                float b = (sin((x + y) * 0.1f) * 0.5f + 0.5f) * 255;

                defaultData[index] = static_cast<unsigned char>(r);
                defaultData[index + 1] = static_cast<unsigned char>(g);
                defaultData[index + 2] = static_cast<unsigned char>(b);
            }
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSize, texSize, 0, GL_RGB, GL_UNSIGNED_BYTE, defaultData.data());
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        std::cout << "Created default colorful texture" << std::endl;
    }

    void calculateModelDimensions() {
        if (meshes.empty()) return;

        glm::vec3 minBounds(FLT_MAX);
        glm::vec3 maxBounds(-FLT_MAX);

        for (auto& mesh : meshes) {
            for (auto& vertex : mesh.vertices) {
                minBounds = glm::min(minBounds, vertex.Position);
                maxBounds = glm::max(maxBounds, vertex.Position);
            }
        }

        modelSize = maxBounds - minBounds;
        modelCenter = (minBounds + maxBounds) * 0.5f;

        std::cout << "Model dimensions: " << modelSize.x << " x " << modelSize.y << " x " << modelSize.z << std::endl;
        std::cout << "Model center: (" << modelCenter.x << ", " << modelCenter.y << ", " << modelCenter.z << ")" << std::endl;
        std::cout << "Recommended scale factor: " << getScaleFactor() << std::endl;
    }

    glm::mat4 aiMatrixToGlm(const aiMatrix4x4& from) {
        glm::mat4 to;
        to[0][0] = from.a1; to[0][1] = from.b1; to[0][2] = from.c1; to[0][3] = from.d1;
        to[1][0] = from.a2; to[1][1] = from.b2; to[1][2] = from.c2; to[1][3] = from.d2;
        to[2][0] = from.a3; to[2][1] = from.b3; to[2][2] = from.c3; to[2][3] = from.d3;
        to[3][0] = from.a4; to[3][1] = from.b4; to[3][2] = from.c4; to[3][3] = from.d4;
        return to;
    }
};

class Animator {
private:
    std::vector<glm::mat4> m_FinalBoneMatrices;
    float m_CurrentTime;
    float m_DeltaTime;
    int m_CurrentAnimationIndex;
    int m_DefaultAnimationIndex;
    int m_RequestedAnimationIndex;
    bool m_IsPlayingTemporaryAnimation;
    float m_TemporaryAnimationDuration;
    float m_TemporaryAnimationTimer;

public:
    Animator() {
        m_FinalBoneMatrices.resize(100, glm::mat4(1.0f));
        m_CurrentTime = 0.0f;
        m_DeltaTime = 0.0f;
        m_CurrentAnimationIndex = 0;
        m_DefaultAnimationIndex = 0;
        m_RequestedAnimationIndex = 0;
        m_IsPlayingTemporaryAnimation = false;
        m_TemporaryAnimationDuration = 0.0f;
        m_TemporaryAnimationTimer = 0.0f;
    }

    void UpdateAnimation(float dt, Model& model) {
        if (!model.HasAnimations()) return;

        m_DeltaTime = dt;

        // Handle temporary animation timing
        if (m_IsPlayingTemporaryAnimation) {
            m_TemporaryAnimationTimer -= dt;

            // If temporary animation finished, return to default
            if (m_TemporaryAnimationTimer <= 0.0f) {
                m_IsPlayingTemporaryAnimation = false;
                m_CurrentAnimationIndex = m_DefaultAnimationIndex;
                m_CurrentTime = 0.0f; // Reset time for default animation
                std::cout << "Returned to default animation: " << model.GetAnimationName(m_DefaultAnimationIndex) << std::endl;
            }
        }

        // Get the current animation (either temporary or default)
        Animation* currentAnimation = model.animations[m_CurrentAnimationIndex].get();
        m_CurrentTime += currentAnimation->m_TicksPerSecond * m_DeltaTime;
        m_CurrentTime = fmod(m_CurrentTime, currentAnimation->m_Duration);

        CalculateBoneTransform(model, model.m_Scene->mRootNode, glm::mat4(1.0f), *currentAnimation);
    }

    // Play an animation temporarily, then return to default
    void PlayTemporaryAnimation(int index, Model& model, float duration = 0.0f) {
        if (index >= 0 && index < model.GetAnimationCount() && index != m_CurrentAnimationIndex) {
            m_RequestedAnimationIndex = index;
            m_IsPlayingTemporaryAnimation = true;

            // If duration is 0, play one full cycle
            if (duration <= 0.0f) {
                Animation* anim = model.animations[index].get();
                duration = anim->m_Duration / anim->m_TicksPerSecond;
            }

            m_TemporaryAnimationDuration = duration;
            m_TemporaryAnimationTimer = duration;
            m_CurrentAnimationIndex = index;
            m_CurrentTime = 0.0f;

            std::cout << "Playing temporary animation: " << model.GetAnimationName(index)
                << " for " << duration << " seconds" << std::endl;
        }
    }

    // Set permanent animation (becomes the new default)
    void SetAnimation(int index, Model& model) {
        if (index >= 0 && index < model.GetAnimationCount()) {
            m_CurrentAnimationIndex = index;
            m_DefaultAnimationIndex = index;
            m_IsPlayingTemporaryAnimation = false;
            m_CurrentTime = 0.0f;
            std::cout << "Set new default animation: " << model.GetAnimationName(index) << std::endl;
        }
    }

    // Set default animation without interrupting current temporary animation
    void SetDefaultAnimation(int index, Model& model) {
        if (index >= 0 && index < model.GetAnimationCount()) {
            m_DefaultAnimationIndex = index;
            std::cout << "Default animation set to: " << model.GetAnimationName(index)
                << " (will activate after current animation)" << std::endl;
        }
    }

    // Next animation (temporary)
    void NextAnimation(Model& model) {
        int nextIndex = (m_CurrentAnimationIndex + 1) % model.GetAnimationCount();
        PlayTemporaryAnimation(nextIndex, model);
    }

    // Previous animation (temporary)
    void PreviousAnimation(Model& model) {
        int prevIndex = (m_CurrentAnimationIndex - 1 + model.GetAnimationCount()) % model.GetAnimationCount();
        PlayTemporaryAnimation(prevIndex, model);
    }

    // Force return to default animation immediately
    void ReturnToDefault(Model& model) {
        if (m_CurrentAnimationIndex != m_DefaultAnimationIndex) {
            m_IsPlayingTemporaryAnimation = false;
            m_CurrentAnimationIndex = m_DefaultAnimationIndex;
            m_CurrentTime = 0.0f;
            std::cout << "Returned to default animation: " << model.GetAnimationName(m_DefaultAnimationIndex) << std::endl;
        }
    }

    int GetCurrentAnimationIndex() const {
        return m_CurrentAnimationIndex;
    }

    int GetDefaultAnimationIndex() const {
        return m_DefaultAnimationIndex;
    }

    bool IsPlayingTemporaryAnimation() const {
        return m_IsPlayingTemporaryAnimation;
    }

    float GetRemainingTemporaryAnimationTime() const {
        return m_TemporaryAnimationTimer;
    }

    const std::vector<glm::mat4>& GetFinalBoneMatrices() const {
        return m_FinalBoneMatrices;
    }

private:
    void CalculateBoneTransform(Model& model, const aiNode* node, const glm::mat4& parentTransform, Animation& animation) {
        std::string nodeName = node->mName.C_Str();
        glm::mat4 nodeTransform = aiMatrixToGlm(node->mTransformation);

        Bone* bone = animation.FindBone(nodeName);

        if (bone) {
            bone->Update(m_CurrentTime);
            nodeTransform = bone->GetLocalTransform();
        }

        glm::mat4 globalTransformation = parentTransform * nodeTransform;

        if (model.boneInfoMap.find(nodeName) != model.boneInfoMap.end()) {
            int index = model.boneInfoMap[nodeName].id;
            m_FinalBoneMatrices[index] = globalTransformation * model.boneInfoMap[nodeName].offset;
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            CalculateBoneTransform(model, node->mChildren[i], globalTransformation, animation);
        }
    }

    glm::mat4 aiMatrixToGlm(const aiMatrix4x4& from) {
        glm::mat4 to;
        to[0][0] = from.a1; to[0][1] = from.b1; to[0][2] = from.c1; to[0][3] = from.d1;
        to[1][0] = from.a2; to[1][1] = from.b2; to[1][2] = from.c2; to[1][3] = from.d2;
        to[2][0] = from.a3; to[2][1] = from.b3; to[2][2] = from.c3; to[2][3] = from.d3;
        to[3][0] = from.a4; to[3][1] = from.b4; to[3][2] = from.c4; to[3][3] = from.d4;
        return to;
    }
};

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

// Settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// Camera
glm::vec3 cameraPos = glm::vec3(0.0f, 1.0f, 5.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Mouse input
bool firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float yaw = -90.0f;
float pitch = 0.0f;

// Ground level
const float GROUND_LEVEL = 0.0f;  // Ground is at y=0

// Character control
glm::vec3 characterPosition = glm::vec3(0.0f, GROUND_LEVEL, 0.0f); // Start at ground level

float characterRotation = 0.0f;
float targetRotation = 0.0f;
float movementSpeed = 3.0f;
float rotationSpeed = 60.0f;
float rotationLerpSpeed = 12.0f; // Higher = faster response, Lower = more smoothing

// Animation states
bool isMoving = false;
bool wasMoving = false;

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Mixamo FBX Animation - Ground Movement System", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Build and compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Build and compile ground shaders
    unsigned int groundVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(groundVertexShader, 1, &groundVertexShaderSource, NULL);
    glCompileShader(groundVertexShader);

    glGetShaderiv(groundVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(groundVertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::GROUND_SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int groundFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(groundFragmentShader, 1, &groundFragmentShaderSource, NULL);
    glCompileShader(groundFragmentShader);

    glGetShaderiv(groundFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(groundFragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::GROUND_SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link ground shaders
    unsigned int groundShaderProgram = glCreateProgram();
    glAttachShader(groundShaderProgram, groundVertexShader);
    glAttachShader(groundShaderProgram, groundFragmentShader);
    glLinkProgram(groundShaderProgram);

    glGetProgramiv(groundShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(groundShaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::GROUND_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(groundVertexShader);
    glDeleteShader(groundFragmentShader);

    // Load your Mixamo character model
    std::cout << "Loading Mixamo Character Model..." << std::endl;

    std::string modelPath = "models/fun_boy2.fbx";

    Model character(modelPath);

    std::cout << "Model loaded with " << character.meshes.size() << " meshes and "
        << character.boneInfoMap.size() << " bones" << std::endl;

    // Print available animations
    std::cout << "\n=== AVAILABLE ANIMATIONS ===" << std::endl;
    for (int i = 0; i < character.GetAnimationCount(); i++) {
        std::cout << i << ": " << character.GetAnimationName(i) << std::endl;
    }
    std::cout << "============================\n" << std::endl;

    // Create ground
    Ground ground;

    Animator animator;

    // Set default animation to idle
    int idleAnimIndex = 8; // idle animation
    int walkAnimIndex = 3; // walk animation
    animator.SetAnimation(idleAnimIndex, character);

    // Configure global OpenGL state
    glEnable(GL_DEPTH_TEST);

    // Calculate automatic scaling
    float autoScale = character.getScaleFactor();
    std::cout << "Using auto-scale factor: " << autoScale << std::endl;

    // Print enhanced controls
    std::cout << "\n=== ENHANCED ANIMATION CONTROLS ===" << std::endl;
    std::cout << "W: Move forward with walk animation" << std::endl;
    std::cout << "Release W: Stop moving with idle animation" << std::endl;
    std::cout << "A/D: Rotate character" << std::endl;
    std::cout << "Arrow Keys: Move camera" << std::endl;
    std::cout << "Mouse: Look around" << std::endl;
    std::cout << "1-9: Play animation temporarily (returns to default)" << std::endl;
    std::cout << "0: Set new default animation (current animation)" << std::endl;
    std::cout << "N: Next temporary animation" << std::endl;
    std::cout << "P: Previous temporary animation" << std::endl;
    std::cout << "R: Return to default animation immediately" << std::endl;
    std::cout << "ESC: Exit" << std::endl;
    std::cout << "==================================\n" << std::endl;

    // Variables for animation info display
    float infoDisplayTimer = 0.0f;
    const float INFO_DISPLAY_TIME = 3.0f;
    std::string currentAnimationInfo = "";

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);

        // Handle movement-based animation transitions
        wasMoving = isMoving;

        // Check if W key is pressed for movement
        isMoving = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS);

        // Animation transition logic
        if (isMoving && !wasMoving) {
            // Started moving - switch to walk animation
            animator.SetAnimation(walkAnimIndex, character);
            std::cout << "Started moving - Walk animation" << std::endl;
        }
        else if (!isMoving && wasMoving) {
            // Stopped moving - switch to idle animation
            animator.SetAnimation(idleAnimIndex, character);
            std::cout << "Stopped moving - Idle animation" << std::endl;
        }

        // Animation controls - TEMPORARY animations (return to default)
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(0, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(0);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(1, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(1);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(2, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(2);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(3, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(3);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(4, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(4);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(5, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(5);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(6, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(6);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(7, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(7);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }
        if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS) {
            animator.PlayTemporaryAnimation(8, character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(8);
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }

        // Set NEW DEFAULT animation (key 0)
        if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS) {
            animator.SetAnimation(animator.GetCurrentAnimationIndex(), character);
            currentAnimationInfo = "New Default: " + character.GetAnimationName(animator.GetCurrentAnimationIndex());
            infoDisplayTimer = INFO_DISPLAY_TIME;
        }

        // Temporary animation navigation
        static bool nKeyPressed = false;
        static bool pKeyPressed = false;
        static bool rKeyPressed = false;

        if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS && !nKeyPressed) {
            animator.NextAnimation(character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(animator.GetCurrentAnimationIndex());
            infoDisplayTimer = INFO_DISPLAY_TIME;
            nKeyPressed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_N) == GLFW_RELEASE) {
            nKeyPressed = false;
        }

        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !pKeyPressed) {
            animator.PreviousAnimation(character);
            currentAnimationInfo = "Playing: " + character.GetAnimationName(animator.GetCurrentAnimationIndex());
            infoDisplayTimer = INFO_DISPLAY_TIME;
            pKeyPressed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE) {
            pKeyPressed = false;
        }

        // Force return to default
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rKeyPressed) {
            animator.ReturnToDefault(character);
            currentAnimationInfo = "Returned to: " + character.GetAnimationName(animator.GetDefaultAnimationIndex());
            infoDisplayTimer = INFO_DISPLAY_TIME;
            rKeyPressed = true;
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE) {
            rKeyPressed = false;
        }

        // Update animation
        if (character.HasAnimations()) {
            animator.UpdateAnimation(deltaTime, character);
        }

        // Update smooth rotation interpolation 
        float rotationDelta = targetRotation - characterRotation;

        // Normalize the difference to always take the shortest path
        if (rotationDelta > 180.0f) {
            rotationDelta -= 360.0f;
        }
        else if (rotationDelta < -180.0f) {
            rotationDelta += 360.0f;
        }

        float t = 1.0f - glm::exp(-rotationLerpSpeed * deltaTime);
        characterRotation += rotationDelta * t;

        // Keep within 0-360 range
        characterRotation = fmod(characterRotation, 360.0f);
        if (characterRotation < 0.0f) characterRotation += 360.0f;

        // Snap to target when very close to prevent jitter
        if (glm::abs(targetRotation - characterRotation) < 0.1f) {
            characterRotation = targetRotation;
        }

        // Update info display timer
        if (infoDisplayTimer > 0.0f) {
            infoDisplayTimer -= deltaTime;
        }

        // Render
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set up view and projection matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        // Draw ground first
        glUseProgram(groundShaderProgram);
        glm::mat4 groundModel = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(groundShaderProgram, "projection"), 1, GL_FALSE,
            glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(groundShaderProgram, "view"), 1, GL_FALSE,
            glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(groundShaderProgram, "model"), 1, GL_FALSE,
            glm::value_ptr(groundModel));
        glUniform3fv(glGetUniformLocation(groundShaderProgram, "lightPos"), 1, glm::value_ptr(glm::vec3(0.0f, 5.0f, 0.0f)));
        glUniform3fv(glGetUniformLocation(groundShaderProgram, "lightColor"), 1, glm::value_ptr(glm::vec3(1.0f, 1.0f, 1.0f)));
        glUniform3fv(glGetUniformLocation(groundShaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));
        ground.Draw(groundShaderProgram);

        // Draw character
        glUseProgram(shaderProgram);

        // Create model matrix with character control
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, characterPosition);
        model = glm::scale(model, glm::vec3(autoScale));
        model = glm::rotate(model, glm::radians(characterRotation), glm::vec3(0.0f, 1.0f, 0.0f));
        //model = glm::translate(model, -character.getCenter()); // Center the model

        // Set matrices
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE,
            glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE,
            glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE,
            glm::value_ptr(model));

        // Set view position for specular lighting
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPos));

        // Set bone matrices
        const auto& boneMatrices = animator.GetFinalBoneMatrices();
        for (int i = 0; i < boneMatrices.size(); i++) {
            std::string name = "finalBonesMatrices[" + std::to_string(i) + "]";
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, name.c_str()), 1, GL_FALSE,
                glm::value_ptr(boneMatrices[i]));
        }

        // Draw character
        character.Draw(shaderProgram);

        // Display current animation info in console
        static int lastAnimIndex = -1;
        static bool lastWasTemporary = false;
        if (animator.GetCurrentAnimationIndex() != lastAnimIndex ||
            animator.IsPlayingTemporaryAnimation() != lastWasTemporary) {

            lastAnimIndex = animator.GetCurrentAnimationIndex();
            lastWasTemporary = animator.IsPlayingTemporaryAnimation();

            std::cout << "Current Animation: " << character.GetAnimationName(lastAnimIndex);
            if (animator.IsPlayingTemporaryAnimation()) {
                std::cout << " (Temporary - " << animator.GetRemainingTemporaryAnimationTime() << "s remaining)";
            }
            else {
                std::cout << " (Default)";
            }
            std::cout << std::endl;
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Character movement
    float cameraSpeed = movementSpeed * deltaTime;
    float rotationSpeedValue = rotationSpeed * deltaTime;

    // Store movement direction
    glm::vec3 moveDirection = glm::vec3(0.0f);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        moveDirection += glm::vec3(sin(glm::radians(characterRotation)), 0.0f, cos(glm::radians(characterRotation)));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        moveDirection -= glm::vec3(sin(glm::radians(characterRotation)), 0.0f, cos(glm::radians(characterRotation)));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        targetRotation += rotationSpeedValue;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        targetRotation -= rotationSpeedValue;
    }

    // Only move if W is pressed (forward movement)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        moveDirection = glm::normalize(moveDirection);
        characterPosition += moveDirection * cameraSpeed;

        // Keep character on ground
        characterPosition.y = GROUND_LEVEL;
    }

    targetRotation = fmod(targetRotation, 360.0f);
    if (targetRotation < 0.0f) targetRotation += 360.0f;

    // Camera movement (free look)
    float cameraMoveSpeed = 5.0f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        cameraPos += cameraMoveSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cameraPos -= cameraMoveSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMoveSpeed;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraMoveSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Optional: implement zoom if needed
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}