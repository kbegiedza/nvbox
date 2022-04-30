#ifndef DEVICE_STATUS_SERVICE_HPP_
#define DEVICE_STATUS_SERVICE_HPP_

#include <memory>
#include <vector>
#include <nvml.h>
#include <cstring>

#include "IService.hpp"
#include "DeviceStatusModel.hpp"

namespace nvbox
{
    class DeviceStatusService : public IService
    {
    public:
        DeviceStatusService(std::shared_ptr<DeviceStatusModel> model)
            : _model(model)
        {
            auto deviceCount = GetDeviceCount();

            if (deviceCount <= 0)
            {
                std::cerr << "Unable to find any device"
                          << std::endl;

                return;
            }

            for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId)
            {
                nvmlDevice_t device;
                auto operationStatus = nvmlDeviceGetHandleByIndex_v2(deviceId, &device);

                if (operationStatus == nvmlReturn_t::NVML_SUCCESS)
                {
                    auto status = DiscoverDeviceStatus(device);

                    if (status != nullptr)
                    {
                        _model->DeviceStatuses.push_back(std::move(*status));
                    }
                }

                if (operationStatus != nvmlReturn_t::NVML_SUCCESS)
                {
                    std::cerr << "Unable to get handle to device"
                              << deviceId
                              << std::endl
                              << operationStatus
                              << std::endl;
                }
            }
        }

        virtual ~DeviceStatusService()
        {
        }

        void Update() override
        {
            for (auto &&deviceStatus : _model->DeviceStatuses)
            {
                deviceStatus.Temperature = GetDeviceTemerature(deviceStatus.GetDevice());
            }
        }

    private:
        const uint32_t GetDeviceTemerature(const nvmlDevice_t &device) const
        {
            uint32_t result = 0;

            nvmlDeviceGetTemperature(device, nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU, &result);

            return result;
        }

        const uint32_t GetDeviceCount() const
        {
            uint32_t deviceCount = 0;

            auto operationStatus = nvmlDeviceGetCount_v2(&deviceCount);

            if (operationStatus != nvmlReturn_t::NVML_SUCCESS)
            {
                std::cerr << "Unable to query device count\n"
                          << operationStatus
                          << std::endl;
            }

            return deviceCount;
        }

        DeviceStatus *DiscoverDeviceStatus(const nvmlDevice_t &device) const
        {
            char *uuid = new char[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
            auto operationStatus = nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_V2_BUFFER_SIZE);

            if (operationStatus == nvmlReturn_t::NVML_SUCCESS)
            {
                auto status = new DeviceStatus(device, uuid);

                status->Temperature = GetDeviceTemerature(device);

                return status;
            }

            delete[] uuid;

            return nullptr;
        }

    private:
        std::shared_ptr<DeviceStatusModel> _model;
    };
}

#endif //! DEVICE_STATUS_SERVICE_HPP_