#include "nn-vulkan.hpp"

#define DEBUG_VULKAN_BUFFERS false
#define DEBUG_VULKAN_TRACE false

#if DEBUG_VULKAN_TRACE
    #define VULKAN_TRACE(...) printf("VULKAN_TRACE: "); printf(__VA_ARGS__); printf("\n");
#else
    #define VULKAN_TRACE(...)
#endif

static bool hasPortabilityExtension() {
#ifdef __APPLE__
    const std::vector<vk::ExtensionProperties> extensionProperties = vk::enumerateInstanceExtensionProperties();
    for (const auto& extension : extensionProperties) {
        if (std::strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0)
            return true;
    }
#endif
    return false;
}

static bool hasValidationLayer() {
    const std::vector<vk::LayerProperties> layerProperties = vk::enumerateInstanceLayerProperties();
    for (const auto& layer : layerProperties) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            return true;
    }
    return false;
}

static void assertDeviceExtensionSupport(const vk::PhysicalDevice &physicalDevice, const std::vector<const char *> &requiredExtensions) {
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    for (const auto& ext : requiredExtensions) {
        bool found = false;
        for (const auto& extension : availableExtensions) {
            if (std::strcmp(extension.extensionName, ext) == 0) {
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("Device extension ") + ext + " is not supported");
    }
}

#define MEMORY_TYPE_INDEX_NOT_FOUND ~0

static uint32_t findMemoryTypeIndex(const vk::PhysicalDevice *physicalDevice, vk::MemoryPropertyFlags expectedFlags) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice->getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; index++) {
        auto flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & expectedFlags) == expectedFlags) {
            return index;
        }
    }
    return MEMORY_TYPE_INDEX_NOT_FOUND;
}

NnVulkanContext::NnVulkanContext(const NnUint gpuIndex) {
    vk::InstanceCreateFlags createInstanceFlags(0);
    std::vector<const char*> instanceLayers = {};
    std::vector<const char*> instanceExtensions = {};
    std::vector<const char*> deviceExtension = {};

    if (hasValidationLayer()) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    }
    if (hasPortabilityExtension()) {
        createInstanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        instanceExtensions.push_back("VK_KHR_portability_enumeration");
        deviceExtension.push_back("VK_KHR_portability_subset");
    }
    deviceExtension.push_back("VK_KHR_8bit_storage");
    deviceExtension.push_back("VK_KHR_16bit_storage");
    deviceExtension.push_back("VK_KHR_maintenance4");

    vk::ApplicationInfo appInfo {"Distributed Llama", 1, nullptr, 0, VK_API_VERSION_1_2 };
    vk::InstanceCreateInfo instanceCreateInfo(createInstanceFlags, &appInfo, instanceLayers, instanceExtensions);
    instance = vk::createInstance(instanceCreateInfo);

    auto physicalDevices = instance.enumeratePhysicalDevices();
    const NnSize nDevices = physicalDevices.size();
    if (gpuIndex >= nDevices)
        throw std::runtime_error("Invalid GPU index, found " + std::to_string(nDevices) + " GPUs");
    physicalDevice = physicalDevices[gpuIndex];
    assertDeviceExtensionSupport(physicalDevice, deviceExtension);

    vk::PhysicalDeviceProperties deviceProps = physicalDevice.getProperties();
    printf("🌋 Device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 DeviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));
    printf("🌋 MaxComputeSharedMemory: %d kB\n", deviceProps.limits.maxComputeSharedMemorySize / 1024);
    printf("🌋 NonCoherentAtomSize: %lu bytes\n", (NnSize)deviceProps.limits.nonCoherentAtomSize);

    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (unsigned int h = 0; h < memoryProperties.memoryHeapCount; h++) {
        if (memoryProperties.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            printf("🌋 Heap[%u]: %lu MB\n", h, ((NnSize)memoryProperties.memoryHeaps[h].size) / (1024 * 1024));
    }

    vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();
    VkPhysicalDeviceFeatures2 deviceFeatures2;
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = nullptr;
    deviceFeatures2.features = (VkPhysicalDeviceFeatures)deviceFeatures;

    VkPhysicalDeviceVulkan11Features vk11Features;
    vk11Features.pNext = nullptr;
    vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    deviceFeatures2.pNext = &vk11Features;

    VkPhysicalDeviceVulkan12Features vk12Features;
    vk12Features.pNext = nullptr;
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11Features.pNext = &vk12Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);

    std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

    const float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
    deviceCreateInfo.enabledExtensionCount = deviceExtension.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtension.data();
    deviceCreateInfo.setPNext(&deviceFeatures2);
    device = physicalDevice.createDevice(deviceCreateInfo);

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex);
    commandPool = device.createCommandPool(commandPoolCreateInfo);
    queue = device.getQueue(queueFamilyIndex, 0);
    nonCoherentAtomSize = deviceProps.limits.nonCoherentAtomSize;
    VULKAN_TRACE("Context created");
}

NnVulkanContext::~NnVulkanContext() {
    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
    VULKAN_TRACE("Context destroyed");
}

std::pair<vk::Buffer, vk::DeviceMemory> NnVulkanContext::createRawBuffer(const uint32_t memoryTypeIndex, const vk::DeviceSize bufferSize, const vk::BufferUsageFlags usageFlags) {
    vk::BufferCreateInfo bufferCreateInfo {
        vk::BufferCreateFlags(),
        bufferSize,
        usageFlags,
        vk::SharingMode::eExclusive,
        1,
        &queueFamilyIndex
    };
    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo bufferMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
    vk::DeviceMemory bufferMemory = device.allocateMemory(bufferMemoryAllocateInfo);

    device.bindBufferMemory(buffer, bufferMemory, 0);
    return std::make_pair(buffer, bufferMemory);
}

NnVulkanStagingCopier::NnVulkanStagingCopier(NnVulkanContext *context) :
    context(context)
{
    allocatedSize = 0u;

    memoryTypeIndex = findMemoryTypeIndex(&context->physicalDevice, vk::MemoryPropertyFlagBits::eHostVisible);
    if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
        throw std::runtime_error("Cannot find host visible memory type");
}

NnVulkanStagingCopier::~NnVulkanStagingCopier() {
    tryRelease();
}

void NnVulkanStagingCopier::tryRelease() {
    if (allocatedSize > 0u) {
        context->device.unmapMemory(hostMemory);
        context->device.freeMemory(hostMemory);
        context->device.destroyBuffer(hostBuffer);
        allocatedSize = 0u;
    }
}

void NnVulkanStagingCopier::allocate(const NnSize size) {
    if (allocatedSize != size) {
        tryRelease();

        auto b = context->createRawBuffer(memoryTypeIndex, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
        hostBuffer = b.first;
        hostMemory = b.second;
        hostPointer = context->device.mapMemory(hostMemory, 0, size);
        allocatedSize = size;
    }
}

void NnVulkanStagingCopier::copy(NnByte *data, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    assert(size == allocatedSize);

    switch (direction) {
    case COPY_TO_DEVICE:
        std::memcpy(hostPointer, data, size);
        break;
    case COPY_FROM_DEVICE:
        std::memcpy(data, hostPointer, size);
        break;
    }
}

void NnVulkanStagingCopier::addCopyCommand(vk::CommandBuffer& commandBuffer, vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    assert(size == allocatedSize);

    VkBufferCopy copyRegion;
	copyRegion.size = size;
    copyRegion.srcOffset = 0u;
    copyRegion.dstOffset = offset;
    switch (direction) {
    case COPY_TO_DEVICE:
        vkCmdCopyBuffer(commandBuffer, hostBuffer, target, 1, &copyRegion);
        break;
    case COPY_FROM_DEVICE:
        vkCmdCopyBuffer(commandBuffer, target, hostBuffer, 1, &copyRegion);
        break;
    }
}

void NnVulkanStagingCopier::executeCopyCommand(vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction) {
    vk::CommandBufferAllocateInfo allocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(allocInfo);
    vk::CommandBuffer commandBuffer = cmdBuffers.front();
    commandBuffer.begin({ vk::CommandBufferUsageFlags{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } });
    addCopyCommand(commandBuffer, target, offset, size, direction);
    commandBuffer.end();

    vk::Fence fence = context->device.createFence(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.destroyFence(fence);
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
}

NnVulkanBuffer::NnVulkanBuffer(NnVulkanContext *context, NnVulkanStagingCopier *copier, const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess) :
    context(context),
    copier(copier),
    name(name),
    bufferSize(bufferSize),
    isSliceable(isSliceable),
    usageFlags(usageFlags)
{
    this->hostPointer = nullptr;

    isHostVisible = false;

    VULKAN_TRACE("Creating buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);

    uint32_t memoryTypeIndex = MEMORY_TYPE_INDEX_NOT_FOUND;
    if (fastAccess) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostCached
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            memoryTypeIndex = findMemoryTypeIndex(
                &context->physicalDevice,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            );
        if (memoryTypeIndex != MEMORY_TYPE_INDEX_NOT_FOUND)
            isHostVisible = true;
    }
    if (!isHostVisible) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            throw std::runtime_error("Cannot find host visible memory type");
    }

    auto b = context->createRawBuffer(memoryTypeIndex, bufferSize, usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    deviceBuffer = b.first;
    deviceMemory = b.second;
    if (isHostVisible)
        hostPointer = (NnByte *)context->device.mapMemory(deviceMemory, 0, bufferSize);
    VULKAN_TRACE("Created buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);
}

NnVulkanBuffer::~NnVulkanBuffer() {
    if (hostPointer != nullptr)
        context->device.unmapMemory(deviceMemory);
    context->device.freeMemory(deviceMemory);
    context->device.destroyBuffer(deviceBuffer);
    VULKAN_TRACE("Destroyed buffer of size %zu", bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data) {
    write(data, 0u, bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        std::memcpy(&hostPointer[offset], data, size);
        context->device.flushMappedMemoryRanges({ { deviceMemory, offset, (vk::DeviceSize)size } });
        VULKAN_TRACE("Wrote %zu bytes to host visible buffer", size);
    } else {
        copier->allocate(size);
        copier->copy((NnByte *)data, size, COPY_TO_DEVICE);
        copier->executeCopyCommand(deviceBuffer, offset, size, COPY_TO_DEVICE);
        VULKAN_TRACE("Wrote %zu bytes to buffer", size);
    }
}

void NnVulkanBuffer::read(NnByte *data) {
    read(data, 0u, bufferSize);
}

void NnVulkanBuffer::read(NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        context->device.invalidateMappedMemoryRanges({ {deviceMemory, offset, (vk::DeviceSize)size} });
        std::memcpy(data, hostPointer, size);

        VULKAN_TRACE("Read %zu bytes from host visible buffer", size);
    } else {
        copier->allocate(size);
        copier->executeCopyCommand(deviceBuffer, offset, size, COPY_FROM_DEVICE);
        copier->copy(data, size, COPY_FROM_DEVICE);

        VULKAN_TRACE("Read %zu bytes from buffer", size);
    }
}

NnSize NnVulkanBuffer::calcSliceSize(const NnSize nominator, const NnSize denominator) {
    if (!isSliceable)
        return bufferSize;

    assert(bufferSize % denominator == 0);

    NnSize size = (bufferSize / denominator) * nominator;
    if (context->nonCoherentAtomSize != 0) {
        // TODO: this alignment is not needed for coherent memory
        size += context->nonCoherentAtomSize - (size % context->nonCoherentAtomSize);
        size = std::min(size, bufferSize);
    }
    return size;
}

NnVulkanBufferFactory::NnVulkanBufferFactory(NnVulkanContext *context, NnVulkanStagingCopier *copier) :
    context(context),
    copier(copier)
{}

std::unique_ptr<NnVulkanBuffer> NnVulkanBufferFactory::create(const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess) {
    return std::unique_ptr<NnVulkanBuffer>(new NnVulkanBuffer(context, copier, name, bufferSize, isSliceable, usageFlags, fastAccess));
}

NnVulkanDeviceData::NnVulkanDeviceData(NnVulkanBufferFactory *bufferFactory, NnNetConfig *netConfig, NnNodeConfig *nodeConfig) :
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    pipes(netConfig->nPipes),
    buffers(nodeConfig->nBuffers),
    weights(nodeConfig->nWeights),
    internalBuffers()
{
    for (NnUint i = 0; i < netConfig->nPipes; i++) {
        const NnPipeConfig *config = &netConfig->pipes[i];
        const bool isSliceable = config->size.z == 1u;
        pipes[i] = bufferFactory->create(config->name, config->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, true);
    }
    for (NnUint i = 0; i < nodeConfig->nBuffers; i++) {
        const NnBufferConfig *config = &nodeConfig->buffers[i];
        const bool isSliceable = config->size.z == 1u;
        buffers[i] = bufferFactory->create(config->name, config->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, false);
    }
    for (NnUint i = 0; i < nodeConfig->nWeights; ++i) {
        const NnWeightConfig *config = &nodeConfig->weights[i];
        if (config->size.nBytes == 0u)
            throw std::invalid_argument("Shared Vulkan weight cannot be empty");
        weights[i] = bufferFactory->create(config->name, config->size.nBytes, false,
            vk::BufferUsageFlagBits::eStorageBuffer, false);
    }

    std::vector<bool> initializedRopeCaches(nodeConfig->nBuffers, false);
    for (NnUint segmentIndex = 0u; segmentIndex < nodeConfig->nSegments; ++segmentIndex) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        for (NnUint opIndex = 0u; opIndex < segment->nOps; ++opIndex) {
            NnOpConfig *op = &segment->ops[opIndex];
            if (op->code != OP_ROPE)
                continue;
            NnRopeOpConfig *config = (NnRopeOpConfig *)op->config;
            if (config->ropeCacheBufferIndex >= nodeConfig->nBuffers)
                throw std::invalid_argument("RoPE references an invalid Vulkan cache buffer");
            if (initializedRopeCaches[config->ropeCacheBufferIndex])
                continue;
            NnVulkanBuffer *buffer = buffers[config->ropeCacheBufferIndex].get();
            std::vector<NnByte> ropeCache(config->slice.cacheSize.nBytes);
            fullfillRopeCache(config, (float *)ropeCache.data());
            buffer->write(ropeCache.data());
            initializedRopeCaches[config->ropeCacheBufferIndex] = true;
        }
    }
}

NnVulkanDeviceData::~NnVulkanDeviceData() {
    pipes.clear();
    buffers.clear();
    weights.clear();
    internalBuffers.clear();
    VULKAN_TRACE("Destroyed device data");
}

NnSize3D NnVulkanDeviceData::resolveBufferSize(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return nodeConfig->buffers[config->pointerIndex].size;
    if (config->source == SRC_PIPE)
        return netConfig->pipes[config->pointerIndex].size;
    throw std::invalid_argument("Unsupported pointer config");
}

NnVulkanBuffer *NnVulkanDeviceData::resolvePointerVulkanBuffer(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return resolveBufferByIndex(config->pointerIndex);
    if (config->source == SRC_PIPE)
        return resolvePipeByIndex(config->pointerIndex);
    throw std::invalid_argument("Unsupported pointer config");
}

NnUint NnVulkanDeviceData::resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex, NnUint zIndex) {
    assert(batchIndex < netConfig->nBatches);
    if (config->type == PNTR_RAW)
        return 0;

    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;
    const NnUint offsetZ = sizeX * netConfig->nBatches * zIndex;

    if (config->type == PNTR_BATCH)
        return offsetZ + sizeX * batchIndex;
    if (config->type == PNTR_BATCH_Z) {
        if (config->groupIndex >= bufferSize.z || zIndex != 0u)
            throw std::invalid_argument("Invalid Vulkan pointer Z index");
        return sizeX * (netConfig->nBatches * config->groupIndex + batchIndex);
    }
    if (config->type == PNTR_BATCHED_SLICE) {
        assert(sizeX % netConfig->nNodes == 0);
        return offsetZ + sizeX * batchIndex + (sizeX / netConfig->nNodes) * nodeConfig->nodeIndex;
    }
    if (config->type == PNTR_BATCHED_GROUP_SLICE) {
        if (bufferSize.z != 1u || config->groupCount == 0u || config->groupIndex >= config->groupCount ||
            sizeX % (netConfig->nNodes * config->groupCount) != 0u)
            throw std::invalid_argument("Invalid grouped Vulkan batched slice dimensions");
        const NnUint nodeWidth = sizeX / netConfig->nNodes;
        const NnUint groupWidth = nodeWidth / config->groupCount;
        return sizeX * batchIndex + nodeWidth * nodeConfig->nodeIndex + groupWidth * config->groupIndex;
    }
    throw std::runtime_error("Cannot determine buffer offset");
}

NnUint NnVulkanDeviceData::resolveBufferBatchWidth(NnPointerConfig *config) {
    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;

    if (config->type == PNTR_RAW)
        return sizeX;
    if (config->type == PNTR_BATCH)
        return sizeX;
    if (config->type == PNTR_BATCH_Z) {
        if (config->groupIndex >= bufferSize.z)
            throw std::invalid_argument("Invalid Vulkan pointer Z index");
        return sizeX;
    }
    if (config->type == PNTR_BATCHED_SLICE) {
        assert(sizeX % netConfig->nNodes == 0);
        return sizeX / netConfig->nNodes;
    }
    if (config->type == PNTR_BATCHED_GROUP_SLICE) {
        if (bufferSize.z != 1u || config->groupCount == 0u || config->groupIndex >= config->groupCount ||
            sizeX % (netConfig->nNodes * config->groupCount) != 0u)
            throw std::invalid_argument("Invalid grouped Vulkan batched slice dimensions");
        return sizeX / (netConfig->nNodes * config->groupCount);
    }
    throw std::runtime_error("Cannot determine buffer width");
}

NnVulkanBuffer *NnVulkanDeviceData::resolvePipeByIndex(NnUint pipeIndex) {
    assert(pipeIndex < netConfig->nPipes);
    return pipes[pipeIndex].get();
}

NnVulkanBuffer *NnVulkanDeviceData::resolveBufferByIndex(NnUint bufferIndex) {
    assert(bufferIndex < nodeConfig->nBuffers);
    return buffers[bufferIndex].get();
}

NnVulkanBuffer *NnVulkanDeviceData::resolveWeightByIndex(NnUint weightIndex) {
    if (weightIndex >= nodeConfig->nWeights)
        throw std::invalid_argument("Op references an invalid shared Vulkan weight index");
    return weights[weightIndex].get();
}

NnVulkanDevice::NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) :
    context(gpuIndex),
    copier(&context),
    bufferFactory(&context, &copier),
    data(&bufferFactory, netConfig, nodeConfig),
    netConfig(netConfig),
    nodeConfig(nodeConfig),
    netExecution(netExecution)
{}

NnVulkanDevice::~NnVulkanDevice() {}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(&context, &copier, &bufferFactory, &data, netConfig, segmentIndex, segmentConfig, netExecution);
};

void NnVulkanDevice::readBuffer(NnUint bufferIndex, NnByte *destination) {
    data.resolveBufferByIndex(bufferIndex)->read(destination);
}

void NnVulkanDevice::writeBuffer(NnUint bufferIndex, const NnByte *source) {
    data.resolveBufferByIndex(bufferIndex)->write(source);
}

static bool isTiledQ40Embedding(const NnOpConfig *op) {
    return op->code == OP_TIED_EMBEDDING && op->weightSize.floatType == F_Q40 &&
        reinterpret_cast<const NnTiedEmbeddingOpConfig *>(op->config)->mode == TIED_EMBEDDING_PARTIAL;
}

static const char *getShaderFileName(const NnOpConfig *opConfig, const NnOpQuantType quantType) {
    const NnOpCode opCode = opConfig->code;
    if (opCode == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return "merge-add-forward-f32-f32.spv";
        if (quantType == Q80_Q80_F32) return "merge-add-forward-q80-f32.spv";
    }
    if (opCode == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return "merge-sum-forward-f32-f32.spv";
        if (quantType == Q80_Q80_F32) return "merge-sum-forward-q80-f32.spv";
    }
    if (opCode == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return "embedding-forward-f32-f32.spv";
        if (quantType == F32_Q40_F32) return "embedding-forward-f32-q40-f32.spv";
    }
    if (opCode == OP_ROPE) {
        if (quantType == F32_F32_F32) return "rope-forward-f32-f32.spv";
    }
    if (opCode == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return "inv-rms-forward-f32-f32.spv";
    }
    if (opCode == OP_MATMUL) {
        if (quantType == F32_F32_F32) return "matmul-forward-f32-f32-f32.spv";
        if (quantType == Q80_Q40_F32) return "matmul-forward-q80-q40-f32.spv";
    }
    if (opCode == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return "multi-head-att-forward-f32-f32.spv";
    }
    if (opCode == OP_DIFFUSION_ATTENTION) {
        if (quantType == F32_F32_F32) return "diffusion-attention-forward-f32-f32.spv";
    }
    if (opCode == OP_TIED_EMBEDDING) {
        const NnTiedEmbeddingOpConfig *config = (NnTiedEmbeddingOpConfig *)opConfig->config;
        if (config->mode == TIED_EMBEDDING_PARTIAL) {
            if (quantType == F32_F32_F32) return "tied-embedding-partial-forward-f32-f32-f32.spv";
            if (quantType == Q80_Q40_F32) return "tied-embedding-logits-forward-q80-q40-f32.spv";
        }
        if (config->mode == TIED_EMBEDDING_REDUCE) {
            if (quantType == F32_F32_F32) return "tied-embedding-reduce-forward-f32-f32.spv";
        }
        if (config->mode == TIED_EMBEDDING_RESCALE) {
            if (quantType == F32_F32_F32) return "tied-embedding-rescale-forward-f32-f32.spv";
        }
        if (config->mode == TIED_EMBEDDING_LOCAL_MAX || config->mode == TIED_EMBEDDING_GLOBAL_MAX) {
            if (quantType == F32_F32_F32) return "embedding-mix-max-forward-f32-f32.spv";
        }
        if (config->mode == TIED_EMBEDDING_FINALIZE) {
            if (quantType == F32_F32_F32) return "embedding-mix-finalize-forward-f32-f32.spv";
        }
    }
    if (opCode == OP_VOCAB_STATS) {
        if (quantType == F32_F32_F32) return "vocab-stats-forward-f32-f32.spv";
    }
    if (opCode == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) {
            const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)opConfig->config;
            return config->scaleFree == 0u
                ? "rms-norm-forward-f32-f32-f32.spv"
                : "rms-norm-scale-free-forward-f32-f32.spv";
        }
    }
    if (opCode == OP_SILU) {
        if (quantType == F32_F32_F32) return "silu-forward-f32-f32.spv";
    }
    if (opCode == OP_GELU) {
        if (quantType == F32_F32_F32) return "gelu-forward-f32-f32.spv";
    }
    if (opCode == OP_TANH) {
        if (quantType == F32_F32_F32) return "tanh-forward-f32-f32.spv";
    }
    if (opCode == OP_MUL) {
        if (quantType == F32_F32_F32) {
            const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)opConfig->config;
            return config->useWeight == 0u
                ? "mul-forward-f32-f32.spv"
                : "mul-weight-forward-f32-f32-f32.spv";
        }
    }
    if (opCode == OP_SCALE) {
        if (quantType == F32_F32_F32) {
            const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)opConfig->config;
            return config->useScalar == 0u
                ? "scale-forward-f32-f32.spv"
                : "scale-scalar-forward-f32-f32.spv";
        }
    }
    if (opCode == OP_CAST) {
        if (quantType == F32_F32_F32) return "cast-forward-f32-f32.spv";
        if (quantType == F32_F32_Q80) return "cast-forward-f32-q80.spv";
    }
    if (opCode == OP_REPEAT_Z) {
        if (quantType == F32_F32_F32) return "repeatz-forward-f32-f32.spv";
        if (quantType == F32_F32_Q80) return "repeatz-forward-f32-q80.spv";
    }
    if (opCode == OP_SHIFT) {
        if (quantType == F32_F32_F32) return "shift-forward-f32-f32.spv";
    }
    if (opCode == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return "softmax-forward-f32-f32.spv";
    }
    if (opCode == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) {
            const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
            return config->expertScaleFromWeight == 0u
                ? "moe-gate-forward-f32-f32.spv"
                : "moe-gate-weight-forward-f32-f32-f32.spv";
        }
        if (quantType == F32_Q40_F32) throw std::invalid_argument("MOE_GATE expert scale must use F32 weights");
        if (quantType == Q80_F32_F32) throw std::invalid_argument("MOE_GATE requires F32 input");
    }
    throw std::invalid_argument(std::string("Unsupported shader: ") + opCodeToString(opCode) + "/" + opQuantTypeToString(quantType));
}

static void buildShaderLayout(std::vector<NnOpBufferAccess> &a, NnVulkanDeviceData *data, NnVulkanDeviceSegmentData *segmentData, NnUint opIndex, NnOpConfig *opConfig) {
    // input
    a.push_back({ACCESS_READONLY, data->resolvePointerVulkanBuffer(&opConfig->input)});
    // output
    a.push_back({ACCESS_READ_WRITE, data->resolvePointerVulkanBuffer(&opConfig->output)});
    // batch info
    a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpBatchInfoVulkanBuffer(opIndex)});
    // weight
    if (opConfig->weightSize.nBytes > 0)
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpWeightVulkanBuffer(opIndex, opConfig)});
    // config
    if (opConfig->configSize > 0) {
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpConfigVulkanBuffer(opIndex)});

        switch (opConfig->code) {
        case OP_RMS_NORM: {
            const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->invRmsBufferIndex)});
        } break;
        case OP_MUL: {
            const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)opConfig->config;
            if (config->useWeight == 0u)
                a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->multiplierBufferIndex)});
        } break;
        case OP_MATMUL: {
            const NnMatmulOpConfig *config = (NnMatmulOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->activeExpertIndexesBufferIndex)});
        } break;
        case OP_SCALE: {
            const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)opConfig->config;
            if (config->useScalar == 0u)
                a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->scaleBufferIndex)});
        } break;
        case OP_SHIFT: {
            const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->indexPipeIndex)});
        } break;
        case OP_ROPE: {
            const NnRopeOpConfig *config = (NnRopeOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->positionPipeIndex)});
            a.push_back({ACCESS_IMMUTABLE, data->resolveBufferByIndex(config->ropeCacheBufferIndex)});
        } break;
        case OP_MULTIHEAD_ATT: {
            const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->positionPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->queryBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->keyCacheBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->valueCacheBufferIndex)});
            a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->attBufferIndex)});
        } break;
        case OP_DIFFUSION_ATTENTION: {
            const NnDiffusionAttentionOpConfig *config =
                (NnDiffusionAttentionOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->positionPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->prefixLengthPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->currentKeyBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->currentValueBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->keyCacheBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->valueCacheBufferIndex)});
        } break;
        case OP_TIED_EMBEDDING: {
            const NnTiedEmbeddingOpConfig *config = (NnTiedEmbeddingOpConfig *)opConfig->config;
            if (config->mode == TIED_EMBEDDING_PARTIAL) {
                a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->partialBufferIndex)});
                a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->temperaturePipeIndex)});
                if (isTiledQ40Embedding(opConfig))
                    a.push_back({ACCESS_READ_WRITE, segmentData->resolveTiedProbabilityBuffer(opIndex)});
            } else if (config->mode == TIED_EMBEDDING_REDUCE) {
                a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->embeddingMixBufferIndex)});
            } else if (config->mode == TIED_EMBEDDING_RESCALE) {
                a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->vocabularyMaxPipeIndex)});
                a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->globalMaxBufferIndex)});
            } else if (config->mode == TIED_EMBEDDING_LOCAL_MAX || config->mode == TIED_EMBEDDING_GLOBAL_MAX) {
                a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->temperaturePipeIndex)});
            }
        } break;
        case OP_VOCAB_STATS: {
            const NnVocabStatsOpConfig *config = (NnVocabStatsOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->globalMaxBufferIndex)});
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->temperaturePipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->samplingCoinsPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->targetTokensPipeIndex)});
            a.push_back({ACCESS_READONLY, data->resolvePipeByIndex(config->localLogitsPipeIndex)});
        } break;
        case OP_MOE_GATE: {
            const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READ_WRITE, data->resolveBufferByIndex(config->indexesBufferIndex)});
            if (config->expertScaleFromWeight == 0u)
                a.push_back({ACCESS_READONLY, data->resolveBufferByIndex(config->expertScaleBufferIndex)});
        } break;
        default:
            break;
        }
    }
}

static bool containsBuffer(std::vector<NnVulkanBuffer *> &buffers, NnVulkanBuffer *buffer) {
    for (NnVulkanBuffer *b : buffers) {
        if (b == buffer) return true;
    }
    return false;
}

static void resolveBuffersToSync(std::vector<std::vector<NnVulkanBuffer *>>& buffersToSync, std::vector<std::vector<NnOpBufferAccess>>& accesses) {
    std::vector<NnVulkanBuffer *> modifiedBuffers;
    for (NnUint opIndex = 0; opIndex < accesses.size(); opIndex++) {
        if (opIndex > 0) {
            bool flush = false;
            for (const NnOpBufferAccess &access : accesses[opIndex]) {
                if (access.type == ACCESS_READONLY && containsBuffer(modifiedBuffers, access.buffer)) {
                    flush = true;
                    break;
                }
            }
            if (flush) {
                for (NnVulkanBuffer *buffer : modifiedBuffers)
                    buffersToSync[opIndex].push_back(buffer);
                modifiedBuffers.clear();
            }
        }
        for (const NnOpBufferAccess &access : accesses[opIndex]) {
            if (access.type == ACCESS_READ_WRITE && !containsBuffer(modifiedBuffers, access.buffer))
                modifiedBuffers.push_back(access.buffer);
        }
    }
}

static NnUint resolvePointerBatchInfoZ(const NnPointerConfig *pointer, const NnSize3D size) {
    return pointer->type == PNTR_RAW || pointer->type == PNTR_BATCH_Z ? 1u : size.z;
}

static NnUint resolveNumberOfBatchInfoZ(
    const NnOpConfig *opConfig,
    const NnSize3D inputSize,
    const NnSize3D outputSize) {
    return std::max(
        resolvePointerBatchInfoZ(&opConfig->input, inputSize),
        resolvePointerBatchInfoZ(&opConfig->output, outputSize));
}

static std::vector<NnVulkanBatchInfo> buildBatchInfo(NnOpConfig *opConfig, NnVulkanDeviceData *data, NnUint nBatches) {
    const NnUint nZ = resolveNumberOfBatchInfoZ(opConfig,
        data->resolveBufferSize(&opConfig->input),
        data->resolveBufferSize(&opConfig->output)
    );

    std::vector<NnVulkanBatchInfo> batchInfo(nZ * nBatches);
    for (NnUint zIndex = 0; zIndex < nZ; zIndex++) {
        for (NnUint batchIndex = 0; batchIndex < nBatches; batchIndex++) {
            const NnUint b = zIndex * nBatches + batchIndex;
            batchInfo[b].inputOffset = data->resolveBufferBatchOffset(&opConfig->input, batchIndex, zIndex);
            batchInfo[b].inputSizeX = data->resolveBufferBatchWidth(&opConfig->input);
            batchInfo[b].outputOffset = data->resolveBufferBatchOffset(&opConfig->output, batchIndex, zIndex);
            batchInfo[b].outputSizeX = data->resolveBufferBatchWidth(&opConfig->output);
        }
    }
    return batchInfo;
}

static std::vector<NnUint> resolveShaderConstants(const NnOpConfig *opConfig, const NnUint nBatches, const NnSize3D inputSize, const NnSize3D outputSize) {
    std::vector<NnUint> consts;
    consts.push_back(nBatches);

    if (
        opConfig->code == OP_CAST ||
        opConfig->code == OP_MATMUL ||
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_REPEAT_Z ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_GELU ||
        opConfig->code == OP_TANH ||
        opConfig->code == OP_SOFTMAX
    ) {
        const NnUint nZ = resolveNumberOfBatchInfoZ(opConfig, inputSize, outputSize);
        consts.push_back(nZ);
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            constexpr NnUint tileSizeX = 2; // Shader constant
            const NnUint inputBlocks = inputSize.x / Q40_BLOCK_SIZE;
            assert(inputSize.x % Q40_BLOCK_SIZE == 0u);
            consts.push_back((inputBlocks + tileSizeX - 1u) / tileSizeX);
        }
    }
    if (opConfig->code == OP_MOE_GATE) {
        const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
        consts.push_back(inputSize.x);
        consts.push_back(config->k);
    }
    if (opConfig->code == OP_TIED_EMBEDDING) {
        const NnTiedEmbeddingOpConfig *config = (NnTiedEmbeddingOpConfig *)opConfig->config;
        if (config->mode == TIED_EMBEDDING_PARTIAL)
            consts.push_back(config->hiddenSize);
        else if (config->mode == TIED_EMBEDDING_REDUCE)
            consts.push_back(config->tileCount);
    }
    return consts;
}

static NnUint resolveShaderNumberOfWorkGroupsX(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_CAST ||
        opConfig->code == OP_REPEAT_Z
    ) {
        if (outputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE;
        } else {
            constexpr NnUint chunkSize = 4u; // Shader constant
            return outputSize.x / chunkSize;
        }
    }
    if (opConfig->code == OP_MERGE_ADD) {
        if (inputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE; // Yes, outputSize is used here
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            constexpr NnUint tileSizeD = 8u; // Shader constant
            return (opConfig->weightSize.x + tileSizeD - 1u) / tileSizeD;
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MULTIHEAD_ATT)
        return ((NnMultiHeadAttOpConfig *)opConfig->config)->nHeads0;
    if (opConfig->code == OP_DIFFUSION_ATTENTION)
        return ((NnDiffusionAttentionOpConfig *)opConfig->config)->nQueryHeads;
    if (opConfig->code == OP_TIED_EMBEDDING) {
        const NnTiedEmbeddingOpConfig *config = (NnTiedEmbeddingOpConfig *)opConfig->config;
        if (config->mode == TIED_EMBEDDING_PARTIAL)
            return config->tileCount;
        if (config->mode == TIED_EMBEDDING_RESCALE ||
            config->mode == TIED_EMBEDDING_FINALIZE) {
            constexpr NnUint chunkSize = 32u; // Shader local size
            return (outputSize.x + chunkSize - 1u) / chunkSize;
        }
        return 1u;
    }
    if (opConfig->code == OP_VOCAB_STATS) {
        const NnVocabStatsOpConfig *config = (NnVocabStatsOpConfig *)opConfig->config;
        return config->mode == VOCAB_STATS_LOCAL ? config->microshardCount : 1u;
    }
    if (opConfig->code == OP_INV_RMS)
        return ((NnInvRmsOpConfig *)opConfig->config)->nColumns;
    if (opConfig->code == OP_SHIFT) {
        constexpr NnUint chunkSize = 4u; // Shader constant
        return (inputSize.x + chunkSize - 1u) / chunkSize;
    }
    if (
        opConfig->code == OP_EMBEDDING ||
        opConfig->code == OP_RMS_NORM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_GELU ||
        opConfig->code == OP_TANH ||
        opConfig->code == OP_MERGE_SUM
    ) {
        constexpr NnUint chunkSize = 4u; // Shader constant
        assert(outputSize.x % chunkSize == 0);
        return outputSize.x / chunkSize;
    }
    return 1u;
}

static NnUint resolveShaderNumberOfWorkGroupsZ(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_REPEAT_Z ||
        opConfig->code == OP_MOE_GATE
    ) {
        return 1u;
    }
    return resolveNumberOfBatchInfoZ(opConfig, inputSize, outputSize);
}

static std::vector<uint32_t> readShader(const char *fileName) {
    std::vector<uint32_t> code;
    std::string path = std::string("./src/nn/vulkan/") + fileName;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path.c_str(), "rb"), fclose);
    FILE *file = fdPtr.get();
    if (!file)
        throw std::runtime_error("Failed to open shader file: " + path);
    uint32_t chunk[4096];
    for (;;) {
        const size_t bytesRead = fread(chunk, 1u, sizeof(chunk), file);
        if (bytesRead % sizeof(uint32_t) != 0u || ferror(file))
            throw std::runtime_error("Failed to read shader file: " + path);
        code.insert(code.end(), chunk, chunk + bytesRead / sizeof(uint32_t));
        if (bytesRead < sizeof(chunk))
            break;
    }
    if (code.empty())
        throw std::runtime_error("Empty shader file: " + path);
    return code;
}

static vk::DescriptorType toDescriptorType(NnVulkanBuffer *buffer) {
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eUniformBuffer)
        return vk::DescriptorType::eUniformBuffer;
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eStorageBuffer)
        return vk::DescriptorType::eStorageBuffer;
    throw std::invalid_argument("Unsupported buffer usage");
}

NnVulkanDeviceSegmentData::NnVulkanDeviceSegmentData(NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnSegmentConfig *segmentConfig, NnUint nBatches) :
    data(data),
    batchInfoBufferIndex(segmentConfig->nOps, UINT32_MAX),
    weightBufferIndex(segmentConfig->nOps, UINT32_MAX),
    configBufferIndex(segmentConfig->nOps, UINT32_MAX),
    tiedProbabilityBufferIndex(segmentConfig->nOps, UINT32_MAX)
{
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (isTiledQ40Embedding(opConfig)) {
            const NnSize bytes = data->resolvePointerVulkanBuffer(&opConfig->output)->bufferSize;
            data->internalBuffers.push_back(bufferFactory->create("tied_probabilities",
                bytes, true, vk::BufferUsageFlagBits::eStorageBuffer, false));
            tiedProbabilityBufferIndex[opIndex] = data->internalBuffers.size() - 1u;
        }

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, nBatches);
        VULKAN_TRACE("Resolved for %s %zu batch infos", opCodeToString(opConfig->code), batchInfo.size());

        NnSize batchInfoSize = sizeof(NnVulkanBatchInfo) * batchInfo.size();
        data->internalBuffers.push_back(
            bufferFactory->create("batchInfo", batchInfoSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false)
        );
        NnVulkanBuffer *batchInfoBuffer = data->internalBuffers.back().get();
        batchInfoBuffer->write((NnByte *)batchInfo.data());
        batchInfoBufferIndex[opIndex] = data->internalBuffers.size() - 1;

        if (opConfig->weightSize.nBytes > 0 && opConfig->weightIndex == NN_NO_SHARED_WEIGHT) {
            data->internalBuffers.push_back(
                bufferFactory->create("weights", opConfig->weightSize.nBytes, false, vk::BufferUsageFlagBits::eStorageBuffer, false)
            );
            weightBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
        if (opConfig->configSize > 0) {
            data->internalBuffers.push_back(
                bufferFactory->create("config", opConfig->configSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false)
            );
            NnVulkanBuffer *configBuffer = data->internalBuffers.back().get();
            configBuffer->write(opConfig->config);
            configBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
    }
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpBatchInfoVulkanBuffer(NnUint opIndex) {
    assert(batchInfoBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[batchInfoBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpConfigVulkanBuffer(NnUint opIndex) {
    assert(configBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[configBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveTiedProbabilityBuffer(NnUint opIndex) {
    assert(tiedProbabilityBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[tiedProbabilityBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpWeightVulkanBuffer(
    NnUint opIndex,
    const NnOpConfig *opConfig) {
    if (opConfig->weightIndex != NN_NO_SHARED_WEIGHT)
        return data->resolveWeightByIndex(opConfig->weightIndex);
    assert(weightBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[weightBufferIndex[opIndex]].get();
}

NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanStagingCopier *copier, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    context(context),
    copier(copier),
    data(data),
    netConfig(netConfig),
    segmentIndex(segmentIndex),
    segmentConfig(segmentConfig),
    netExecution(netExecution),
    shaderModules(segmentConfig->nOps),
    descriptorSets(segmentConfig->nOps),
    descriptorSetLayouts(segmentConfig->nOps),
    pipelineLayouts(segmentConfig->nOps),
    pipelines(segmentConfig->nOps),
    tiedShaderModules(segmentConfig->nOps),
    tiedPipelines(segmentConfig->nOps),
    buffersToSync(segmentConfig->nOps)
{
    this->segmentData.reset(new NnVulkanDeviceSegmentData(bufferFactory, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfos(segmentConfig->nOps);
    std::vector<std::vector<NnOpBufferAccess>> opBufferAccesses(segmentConfig->nOps);

    std::vector<vk::SpecializationInfo> specInfos(segmentConfig->nOps);
    std::vector<std::vector<vk::SpecializationMapEntry>> specEntries(segmentConfig->nOps);
    std::vector<std::vector<NnUint>> constants(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);
        NnSize3D outputSize = data->resolveBufferSize(&opConfig->output);
        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType
        );
        const char *shaderFileName = getShaderFileName(opConfig, opQuant);
        std::vector<uint32_t> code = readShader(shaderFileName);

        buildShaderLayout(opBufferAccesses[opIndex], data, segmentData.get(), opIndex, opConfig);

        VULKAN_TRACE("Loading shader: %s", shaderFileName);
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            code.size() * sizeof(uint32_t),
            code.data()
        );

        constants[opIndex] = resolveShaderConstants(opConfig, netConfig->nBatches, inputSize, outputSize);
        const std::vector<NnUint> *opConstants = &constants[opIndex];

        specEntries[opIndex].resize(opConstants->size());
        for (NnUint i = 0; i < opConstants->size(); i++) {
            specEntries[opIndex][i] = vk::SpecializationMapEntry(
                i,
                i * sizeof(NnUint),
                sizeof(NnUint)
            );
            VULKAN_TRACE("Spec constant %d: %u", i, opConstants->at(i));
        }
        specInfos[opIndex] = vk::SpecializationInfo(
            opConstants->size(),
            specEntries[opIndex].data(),
            opConstants->size() * sizeof(NnUint),
            opConstants->data()
        );

        vk::ShaderModule shaderModule = context->device.createShaderModule(shaderModuleCreateInfo);
        vk::PipelineShaderStageCreateInfo shaderCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shaderModule,
            "main",
            &specInfos[opIndex]
        );

        shaderModules[opIndex] = shaderModule;
        shaderCreateInfos[opIndex] = shaderCreateInfo;
        VULKAN_TRACE("Segment %d, opIndex: %d, buffers: %zu", segmentIndex, opIndex, opBufferAccesses.size());
    }

    NnUint nUniformBuffers = 0;
    NnUint nStorageBuffers = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings(accesses.size());
        for (NnUint i = 0; i < accesses.size(); i++) {
            vk::DescriptorType descriptorType = toDescriptorType(accesses[i].buffer);
            descriptorSetLayoutBindings[i] = vk::DescriptorSetLayoutBinding(
                i,
                descriptorType,
                1,
                vk::ShaderStageFlagBits::eCompute
            );
            if (descriptorType == vk::DescriptorType::eUniformBuffer)
                nUniformBuffers++;
            else if (descriptorType == vk::DescriptorType::eStorageBuffer)
                nStorageBuffers++;
        }

        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            descriptorSetLayoutBindings.size(),
            descriptorSetLayoutBindings.data()
        );
        vk::DescriptorSetLayout descriptorSetLayout = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

        descriptorSetLayouts[opIndex] = descriptorSetLayout;
    }

    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    if (nStorageBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, nStorageBuffers));
    if (nUniformBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
        segmentConfig->nOps,
        descriptorPoolSizes.size(),
        descriptorPoolSizes.data()
    );
    descriptorPool = context->device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, descriptorSetLayouts.size(), descriptorSetLayouts.data());
    descriptorSets = context->device.allocateDescriptorSets(descriptorSetAllocInfo);

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets(nUniformBuffers + nStorageBuffers);
    std::vector<vk::DescriptorBufferInfo> bufferInfos(nUniformBuffers + nStorageBuffers);
    NnUint writeDescriptorSetIndex = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        for (NnUint i = 0; i < accesses.size(); i++) {
            NnVulkanBuffer *buffer = accesses[i].buffer;
            bufferInfos[writeDescriptorSetIndex] = vk::DescriptorBufferInfo(
                buffer->deviceBuffer,
                0,
                buffer->bufferSize
            );
            vk::DescriptorType descriptorType = toDescriptorType(buffer);
            writeDescriptorSets[writeDescriptorSetIndex] = vk::WriteDescriptorSet(
                descriptorSets[opIndex],
                i,
                0,
                1,
                descriptorType,
                nullptr,
                &bufferInfos.data()[writeDescriptorSetIndex],
                nullptr
            );
            writeDescriptorSetIndex++;
        }
    }

    context->device.updateDescriptorSets(writeDescriptorSets, nullptr);

    pipelineCache = context->device.createPipelineCache(vk::PipelineCacheCreateInfo());

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayouts[opIndex]);
        const bool tiled = isTiledQ40Embedding(&segmentConfig->ops[opIndex]);
        const vk::PushConstantRange runtimeRange(vk::ShaderStageFlagBits::eCompute, 0u, sizeof(NnUint));
        if (tiled) {
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1u;
            pipelineLayoutCreateInfo.pPushConstantRanges = &runtimeRange;
        }
        pipelineLayouts[opIndex] = context->device.createPipelineLayout(pipelineLayoutCreateInfo);

        vk::ComputePipelineCreateInfo pipelineInfo(vk::PipelineCreateFlags(), shaderCreateInfos[opIndex], pipelineLayouts[opIndex], vk::Pipeline(), 0);
        pipelines[opIndex] = context->device.createComputePipelines(pipelineCache, pipelineInfo).value.front();
        if (tiled) {
            const char *stages[] = {
                "tied-embedding-softmax-forward-q40.spv",
                "tied-embedding-mix-forward-q40.spv",
            };
            for (const char *file : stages) {
                const auto code = readShader(file);
                const auto module = context->device.createShaderModule(vk::ShaderModuleCreateInfo(
                    vk::ShaderModuleCreateFlags(), code.size() * sizeof(uint32_t), code.data()));
                tiedShaderModules[opIndex].push_back(module);
                const vk::PipelineShaderStageCreateInfo stage(vk::PipelineShaderStageCreateFlags(),
                    vk::ShaderStageFlagBits::eCompute, module, "main", &specInfos[opIndex]);
                const vk::ComputePipelineCreateInfo info(vk::PipelineCreateFlags(), stage, pipelineLayouts[opIndex]);
                tiedPipelines[opIndex].push_back(
                    context->device.createComputePipelines(pipelineCache, info).value.front());
            }
        }
    }

    fence = context->device.createFence(vk::FenceCreateInfo());

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(commandBufferAllocInfo);
    commandBuffer = cmdBuffers.front();

    resolveBuffersToSync(buffersToSync, opBufferAccesses);
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
    context->device.destroyDescriptorPool(descriptorPool);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        for (auto pipeline : tiedPipelines[opIndex])
            context->device.destroyPipeline(pipeline);
        for (auto module : tiedShaderModules[opIndex])
            context->device.destroyShaderModule(module);
        context->device.destroyPipeline(pipelines[opIndex]);
        context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }
    context->device.destroyFence(fence);
    context->device.destroyPipelineCache(pipelineCache);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        context->device.destroyShaderModule(shaderModules[opIndex]);
    }
    VULKAN_TRACE("Destroyed segment");
}

void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex < segmentConfig->nOps);
    assert(offset + nBytes <= segmentConfig->ops[opIndex].weightSize.nBytes);
    NnVulkanBuffer *buffer = segmentData->resolveOpWeightVulkanBuffer(opIndex, &segmentConfig->ops[opIndex]);
    buffer->write(weight, offset, nBytes);
}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);

    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    // TODO: this should be called only after weights loading is done
    copier->tryRelease();

    // TODO: refactor this block
    {
        for (NnUint i = 0; i < netConfig->nPreSyncs; i++) {
            NnPreSyncConfig *preSyncConfig = &netConfig->preSyncs[i];
            NnByte *pipeData = netExecution->pipes[preSyncConfig->pipeIndex];
            NnVulkanBuffer *buffer = data->pipes[preSyncConfig->pipeIndex].get();
            const NnUint pipeBatches = netConfig->pipes[preSyncConfig->pipeIndex].size.y;
            buffer->write(pipeData, 0u,
                buffer->calcSliceSize(std::min(batchSize, pipeBatches), pipeBatches));
        }

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->input.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->input.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->input.pointerIndex].get();
                const NnSize transferSize = opConfig->input.type == PNTR_RAW
                    ? buffer->bufferSize
                    : buffer->calcSliceSize(batchSize, netConfig->nBatches);
                buffer->write(pipeData, 0u, transferSize);
            }
        }
    }

    if (lastBatchSize != batchSize) {
        lastBatchSize = batchSize;
        commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            std::vector<vk::BufferMemoryBarrier> memoryBarriers;
            for (NnVulkanBuffer *buffer : buffersToSync[opIndex]) {
                vk::BufferMemoryBarrier barrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead,
                    context->queueFamilyIndex,
                    context->queueFamilyIndex,
                    buffer->deviceBuffer,
                    0,
                    buffer->calcSliceSize(batchSize, netConfig->nBatches)
                );
                memoryBarriers.push_back(barrier);
            }
            if (!memoryBarriers.empty()) {
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(),
                    nullptr,
                    memoryBarriers,
                    nullptr
                );
                VULKAN_TRACE("Created memory barrier for %zu buffers", memoryBarriers.size());
            }

            commandBuffer.bindPipeline(
                vk::PipelineBindPoint::eCompute,
                pipelines[opIndex]
            );
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                pipelineLayouts[opIndex],
                0,
                { descriptorSets[opIndex] },
                {}
            );

            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            const NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);
            const NnSize3D outputSize = data->resolveBufferSize(&opConfig->output);

            if (isTiledQ40Embedding(opConfig)) {
                const auto *config = reinterpret_cast<const NnTiedEmbeddingOpConfig *>(opConfig->config);
                commandBuffer.pushConstants(pipelineLayouts[opIndex], vk::ShaderStageFlagBits::eCompute,
                    0u, sizeof(batchSize), &batchSize);
                const NnUint vocab = data->resolveBufferBatchWidth(&opConfig->output);
                commandBuffer.dispatch((vocab + 7u) / 8u, (batchSize + 3u) / 4u, 1u);
                // Internal dispatches share logits, probabilities and partials.
                // Finish their writes before the next dispatch consumes them.
                const vk::MemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                for (NnUint stage = 0u; stage < 2u; ++stage) {
                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(),
                        {barrier}, {}, {});
                    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, tiedPipelines[opIndex][stage]);
                    if (stage == 0u)
                        commandBuffer.dispatch(config->tileCount, batchSize, 1u);
                    else
                        commandBuffer.dispatch(config->tileCount * (config->hiddenSize / Q40_BLOCK_SIZE),
                            (batchSize + 7u) / 8u, 1u);
                }
                continue;
            }
            const NnUint groupCountX = resolveShaderNumberOfWorkGroupsX(opConfig, inputSize, outputSize);
            const NnUint groupCountY = batchSize;
            const NnUint groupCountZ = resolveShaderNumberOfWorkGroupsZ(opConfig, inputSize, outputSize);
            commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);
            VULKAN_TRACE("Dispatched %s (%d, %d, %d)", opCodeToString(opConfig->code), groupCountX, groupCountY, groupCountZ);
        }
        commandBuffer.end();
    }

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.resetFences({ fence });
    VULKAN_TRACE("Forwarded");

    // TODO: refactor this block
    {
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->output.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->output.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->output.pointerIndex].get();
                const NnSize transferSize = opConfig->output.type == PNTR_RAW
                    ? buffer->bufferSize
                    : buffer->calcSliceSize(batchSize, netConfig->nBatches);
                buffer->read(pipeData, 0u, transferSize);
            }
        }
    }

#if DEBUG_VULKAN_BUFFERS
    NnUint nBuffers = data->buffers.size();
    for (NnUint i = 0; i < nBuffers; i++) {
        NnVulkanBuffer *buffer = data->buffers[i].get();
        printf("[%3d:%3d:%10s] ", segmentIndex, i, buffer->name);
        std::vector<NnByte> data(buffer->bufferSize);
        buffer->read(data.data());
        if (strncmp(buffer->name, "q_", 2) == 0) {
            NnUint nBytes = 32;
            if (buffer->bufferSize < nBytes)
                nBytes = buffer->bufferSize;
            for (NnUint j = 0; j < nBytes; j++)
                printf(" %x", data.data()[j]);
        } else {
            NnUint nNumbers = buffer->bufferSize / sizeof(float);
            if (nNumbers > 16)
                nNumbers = 16;
            float *nums = (float *)data.data();
            for (NnUint j = 0; j < nNumbers; j++)
                printf(" %.4f", nums[j]);
        }
        printf("\n");
    }
#endif
}
