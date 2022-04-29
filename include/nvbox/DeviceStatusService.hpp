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
            : _model(model), _devices()
        {
            uint32_t deviceCount = 0;

            nvmlReturn_t nvmlOpStatus;
            nvmlOpStatus = nvmlDeviceGetCount_v2(&deviceCount);

            if (nvmlOpStatus != nvmlReturn_t::NVML_SUCCESS)
            {
                std::cerr << "Unable to query device count\n"
                          << nvmlOpStatus
                          << std::endl;

                return;
            }

            if (deviceCount <= 0)
            {
                std::cerr << "Unable to find any device"
                          << std::endl;

                return;
            }

            for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId)
            {
                nvmlDevice_t currentDevice;
                nvmlOpStatus = nvmlDeviceGetHandleByIndex_v2(deviceId, &currentDevice);

                if (nvmlOpStatus == nvmlReturn_t::NVML_SUCCESS)
                {
                    _devices.push_back(std::move(currentDevice));
                }
                else
                {
                    std::cerr << "Unable to get handle to device"
                              << deviceId
                              << std::endl
                              << nvmlOpStatus
                              << std::endl;
                }
            }

            for (auto &&device : _devices)
            {
                DeviceStatus status;

                char *uuid = new char[NVML_DEVICE_UUID_V2_BUFFER_SIZE];

                nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_V2_BUFFER_SIZE);

                std::strcpy(status.UUID, uuid);
                delete[] uuid;

                status.Device = device;
                status.Temperature = GetDeviceTemerature(device);

                _model->Devices.push_back(std::move(status));
            }
        }

        virtual ~DeviceStatusService()
        {
        }

        void Update() override
        {
            for (auto &&device : _model->Devices)
            {
                device.Temperature = GetDeviceTemerature(device.Device);
            }
        }

    private:
        uint32_t GetDeviceTemerature(const nvmlDevice_t &device)
        {
            uint32_t result;

            nvmlDeviceGetTemperature(device, nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU, &result);

            return result;
        }

    private:
        std::vector<nvmlDevice_t> _devices;
        std::shared_ptr<DeviceStatusModel> _model;
    };
}

#endif //! DEVICE_STATUS_SERVICE_HPP_