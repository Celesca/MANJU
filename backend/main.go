package main

import (
	"log"

	"github.com/gofiber/fiber/v2"
	"backend/database"
)

func main() {

	// 👉 Connect PostgreSQL
	database.Connect()

	app := fiber.New()

	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"message": "Backend running + DB connected!",
		})
	})

	log.Fatal(app.Listen(":8080"))
}
