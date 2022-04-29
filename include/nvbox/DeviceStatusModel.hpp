#ifndef DEVICE_STATUS_MODEL_HPP_
#define DEVICE_STATUS_MODEL_HPP_

#include <nvml.h>
#include <vector>

namespace nvbox
{
    struct DeviceStatus
    {
    public:
        DeviceStatus()
            : UUID(nullptr)
        {
        }

        ~DeviceStatus()
        {
            if (UUID != nullptr)
            {
                delete[] UUID;
                UUID = nullptr;
            }
        }

    public:
        char *UUID;
        uint32_t Temperature;
        nvmlDevice_t Device;
    };

    struct DeviceStatusModel
    {
    public:
        std::vector<DeviceStatus> Devices;
    };
}

#endif //! DEVICE_STATUS_MODEL_HPP_