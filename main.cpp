#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in ivec4 aBoneIDs;
layout (location = 4) in vec4 aWeights;

out vec2 TexCoords;
out vec3 Normal;
out vec3 FragPos;

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
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos = vec3(0.0, 5.0, 0.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 objectColor = vec3(0.7, 0.5, 0.3);

void main()
{
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

// Bone structure (keep all your existing Bone, Mesh, Animation, Model, Animator classes the same)
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
    unsigned int VAO, VBO, EBO;

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices) {
        this->vertices = vertices;
        this->indices = indices;
        setupMesh();
    }

    void Draw() {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
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

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        glEnableVertexAttribArray(3);
        glVertexAttribIPointer(3, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, BoneIDs));
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Weights));

        glBindVertexArray(0);
    }
};

class Animation {
public:
    float m_Duration;
    float m_TicksPerSecond;
    std::vector<Bone> m_Bones;
    std::map<std::string, BoneInfo> m_BoneInfoMap;

    Animation(const aiScene* scene, const std::string& animationName) {
        aiAnimation* animation = scene->mAnimations[0];
        m_Duration = animation->mDuration;
        m_TicksPerSecond = animation->mTicksPerSecond != 0 ? animation->mTicksPerSecond : 25.0f;

        std::cout << "Loading animation: " << animationName << std::endl;
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
    std::unique_ptr<Animation> animation;

    Model(const std::string& path) {
        loadModel(path);
        boneMatrices.resize(100, glm::mat4(1.0f));
        calculateModelDimensions();
    }

    void Draw() {
        for (auto& mesh : meshes) {
            mesh.Draw();
        }
    }

    float getScaleFactor() {
        float maxDimension = std::max(modelSize.x, std::max(modelSize.y, modelSize.z));
        return 2.0f / maxDimension;
    }

    glm::vec3 getCenter() {
        return modelCenter;
    }

    bool HasAnimation() const {
        return animation != nullptr;
    }
    const aiScene* m_Scene;

private:
    Assimp::Importer m_Importer;

    void loadModel(const std::string& path) {
        m_Scene = m_Importer.ReadFile(path,
            aiProcess_Triangulate |
            aiProcess_FlipUVs |
            aiProcess_CalcTangentSpace |
            aiProcess_GenSmoothNormals);

        if (!m_Scene || m_Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !m_Scene->mRootNode) {
            std::cout << "ERROR::ASSIMP:: " << m_Importer.GetErrorString() << std::endl;
            return;
        }

        if (m_Scene->HasAnimations()) {
            std::cout << "Model has " << m_Scene->mNumAnimations << " animations" << std::endl;
            animation = std::make_unique<Animation>(m_Scene, "MixamoRun");
        }
        else {
            std::cout << "Model has no animations" << std::endl;
        }

        processNode(m_Scene->mRootNode, m_Scene);
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

            vertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
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

        return Mesh(vertices, indices);
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

public:
    Animator() {
        m_FinalBoneMatrices.resize(100, glm::mat4(1.0f));
        m_CurrentTime = 0.0f;
        m_DeltaTime = 0.0f;
    }

    void UpdateAnimation(float dt, Model& model) {
        if (!model.HasAnimation()) return;

        m_DeltaTime = dt;
        m_CurrentTime += model.animation->m_TicksPerSecond * m_DeltaTime;
        m_CurrentTime = fmod(m_CurrentTime, model.animation->m_Duration);

        CalculateBoneTransform(model, model.m_Scene->mRootNode, glm::mat4(1.0f));
    }

    void CalculateBoneTransform(Model& model, const aiNode* node, const glm::mat4& parentTransform) {
        std::string nodeName = node->mName.C_Str();
        glm::mat4 nodeTransform = aiMatrixToGlm(node->mTransformation);

        Bone* bone = model.animation->FindBone(nodeName);

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
            CalculateBoneTransform(model, node->mChildren[i], globalTransformation);
        }
    }

    const std::vector<glm::mat4>& GetFinalBoneMatrices() const {
        return m_FinalBoneMatrices;
    }

private:
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

// Character control
glm::vec3 characterPosition = glm::vec3(0.0f, 0.0f, 0.0f);
float characterRotation = 0.0f;
float movementSpeed = 3.0f;
float rotationSpeed = 60.0f;

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Mixamo FBX Animation - Controlled Movement", NULL, NULL);
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

    // Load your Mixamo character model
    std::cout << "Loading character.fbx..." << std::endl;
    Model character("models/run.fbx");
    std::cout << "Model loaded with " << character.meshes.size() << " meshes and "
        << character.boneInfoMap.size() << " bones" << std::endl;

    Animator animator;

    // Configure global OpenGL state
    glEnable(GL_DEPTH_TEST);

    // Calculate automatic scaling
    float autoScale = character.getScaleFactor();
    std::cout << "Using auto-scale factor: " << autoScale << std::endl;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Input
        processInput(window);

        // Update animation
        if (character.HasAnimation()) {
            animator.UpdateAnimation(deltaTime, character);
        }

        // Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Set up view and projection matrices
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
            (float)SCR_WIDTH / (float)SCR_HEIGHT,
            0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

        // Create model matrix with character control
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, characterPosition);
        model = glm::scale(model, glm::vec3(autoScale));
        model = glm::rotate(model, glm::radians(characterRotation), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::translate(model, -character.getCenter()); // Center the model

        // Set matrices
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE,
            glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE,
            glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE,
            glm::value_ptr(model));

        // Set bone matrices
        const auto& boneMatrices = animator.GetFinalBoneMatrices();
        for (int i = 0; i < boneMatrices.size(); i++) {
            std::string name = "finalBonesMatrices[" + std::to_string(i) + "]";
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, name.c_str()), 1, GL_FALSE,
                glm::value_ptr(boneMatrices[i]));
        }

        // Draw character
        character.Draw();

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
    float rotationSpeedRad = glm::radians(rotationSpeed) * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        characterPosition += cameraSpeed * glm::vec3(sin(glm::radians(characterRotation)), 0.0f, cos(glm::radians(characterRotation)));
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        characterPosition -= cameraSpeed * glm::vec3(sin(glm::radians(characterRotation)), 0.0f, cos(glm::radians(characterRotation)));
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        characterRotation += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        characterRotation -= rotationSpeed;

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