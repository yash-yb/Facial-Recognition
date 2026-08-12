  #ifdef _WIN32
  #include <windows.h>
  #include <mfapi.h>
  #include <mfidl.h>
  #include <mfobjects.h>
  #include <mfreadwrite.h>
  #include <wrl/client.h>
  #include <dshow.h>
    
  #elif __linux__
  #include <linux/videodev2.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <sys/ioctl.h>
  #include <sys/mman.h>
    
  struct Buffer
  {
      void *start;
      size_t length;
  };
    
  #endif

  
int main()
{
    
}