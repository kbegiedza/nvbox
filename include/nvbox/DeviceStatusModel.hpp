#ifndef DEVICE_STATUS_MODEL_HPP_
#define DEVICE_STATUS_MODEL_HPP_

#include <nvml.h>
#include <vector>

namespace nvbox
{
    struct DeviceStatus
    {
    public:
        DeviceStatus(const nvmlDevice_t &device, const char *uuid)
            : _uuid(uuid), _device(device)
        {
        }

        ~DeviceStatus()
        {
            if (_uuid != nullptr)
            {
                delete[] _uuid;
            }
        }

    public:
        const char *const GetUUID() const
        {
            return _uuid;
        }

        const nvmlDevice_t GetDevice() const
        {
            return _device;
        }

    public:
        uint32_t Temperature;

    private:
        const char *const _uuid;
        const nvmlDevice_t _device;
    };

    struct DeviceStatusModel
    {
    public:
        std::vector<DeviceStatus> DeviceStatuses;
    };
}

#endif //! DEVICE_STATUS_MODEL_HPP_