#ifndef ISERVICE_HPP_
#define ISERVICE_HPP_

namespace nvbox
{
    class IService
    {
    protected:
        IService() {}
        virtual ~IService() {}

    public:
        virtual void Update() = 0;
    };
}

#endif //! ISERVICE_HPP_