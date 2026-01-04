using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using App_Tích_hợp_Yolov8_Pose_MediaPipe_Hand.Models;

namespace App_Tích_hợp_Yolov8_Pose_MediaPipe_Hand.Controllers;

public class HomeController : Controller
{
    public IActionResult Index() => View();

    public IActionResult YoloPose() => View();

    public IActionResult MediapipeHand() => View();

    public IActionResult Privacy()
    {
        return View();
    }
}
