package main

import (
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"
)

// Lista de puertos comunes para escanear
var commonPorts = []int{
	20, 21, 22, 23, 25, 53, 80, 88, 110, 111, 135, 139, 143, 
	443, 445, 465, 587, 993, 995, 1433, 1434, 3306, 3389, 5900, 
	5901, 8080, 8443, 8888,
}

// Información básica sobre servicios comunes por puerto
var serviceMap = map[int]string{
	20:   "FTP (data)",
	21:   "FTP (control)",
	22:   "SSH",
	23:   "Telnet",
	25:   "SMTP",
	53:   "DNS",
	80:   "HTTP",
	88:   "Kerberos",
	110:  "POP3",
	111:  "RPC",
	135:  "MSRPC",
	139:  "NetBIOS",
	143:  "IMAP",
	443:  "HTTPS",
	445:  "SMB",
	465:  "SMTPS",
	587:  "SMTP (submission)",
	993:  "IMAPS",
	995:  "POP3S",
	1433: "MSSQL",
	1434: "MSSQL Browser",
	3306: "MySQL",
	3389: "RDP",
	5900: "VNC",
	5901: "VNC",
	8080: "HTTP (alternate)",
	8443: "HTTPS (alternate)",
	8888: "HTTP (alternate)",
}

// Estructura para almacenar resultados
type ScanResult struct {
	Port    int
	Status  string
	Service string
}

func scanPort(ip string, port int, wg *sync.WaitGroup, results chan<- ScanResult) {
	defer wg.Done()
	
	address := ip + ":" + strconv.Itoa(port)
	conn, err := net.DialTimeout("tcp", address, 500*time.Millisecond)
	
	if err != nil {
		// Puerto cerrado o filtrado
		return
	}
	defer conn.Close()
	
	service := "Unknown"
	if serviceName, ok := serviceMap[port]; ok {
		service = serviceName
	}
	
	results <- ScanResult{Port: port, Status: "open", Service: service}
}

func grabBanner(ip string, port int) string {
	address := ip + ":" + strconv.Itoa(port)
	conn, err := net.DialTimeout("tcp", address, 1*time.Second)
	if err != nil {
		return ""
	}
	defer conn.Close()
	
	// Establecer un timeout para la lectura del banner
	conn.SetReadDeadline(time.Now().Add(1 * time.Second))
	
	// Buffer para recibir datos
	buffer := make([]byte, 1024)
	
	// Intentar leer el banner
	_, err = conn.Read(buffer)
	if err != nil {
		return ""
	}
	
	return string(buffer)
}

func main() {
	target := "192.168.1.1"
	fmt.Printf("Iniciando escaneo de puertos en %s\n", target)
	fmt.Println("================================================")
	
	// Canal para recopilar resultados
	results := make(chan ScanResult, len(commonPorts))
	
	// WaitGroup para sincronizar goroutines
	var wg sync.WaitGroup
	
	// Escanear puertos comunes
	for _, port := range commonPorts {
		wg.Add(1)
		go scanPort(target, port, &wg, results)
	}
	
	// Crear una goroutine para cerrar el canal cuando todas las goroutines de escaneo terminen
	go func() {
		wg.Wait()
		close(results)
	}()
	
	// Recopilar resultados
	var openPorts []ScanResult
	for result := range results {
		openPorts = append(openPorts, result)
	}
	
	// Mostrar resultados de forma organizada
	if len(openPorts) == 0 {
		fmt.Println("No se encontraron puertos abiertos.")
	} else {
		fmt.Printf("Encontrados %d puertos abiertos:\n\n", len(openPorts))
		fmt.Printf("%-10s %-15s %s\n", "PUERTO", "SERVICIO", "DETALLES")
		fmt.Println("------------------------------------------")
		
		for _, r := range openPorts {
			banner := grabBanner(target, r.Port)
			bannerInfo := ""
			if banner != "" {
				if len(banner) > 40 {
					bannerInfo = banner[:40] + "..."
				} else {
					bannerInfo = banner
				}
			}
			fmt.Printf("%-10d %-15s %s\n", r.Port, r.Service, bannerInfo)
		}
	}
	
	fmt.Println("\nEscaneo finalizado.")
}
