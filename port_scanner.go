package main

import (
	"fmt"
	"net"
	"sync"
	"time"
	"sort"
	"strings"
)

// Mapeo básico de puertos a servicios comunes
var commonPorts = map[int]string{
	20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet",
	25: "SMTP", 53: "DNS", 80: "HTTP", 110: "POP3",
	115: "SFTP", 135: "RPC", 139: "NetBIOS", 143: "IMAP",
	194: "IRC", 443: "HTTPS", 445: "SMB", 989: "FTPS-data",
	990: "FTPS", 1433: "MSSQL", 3306: "MySQL", 3389: "RDP",
	5432: "PostgreSQL", 5900: "VNC", 6379: "Redis", 8080: "HTTP-Proxy",
	8443: "HTTPS-Alt", 27017: "MongoDB",
}

// Estructura para almacenar resultados
type ScanResult struct {
	Port    int
	State   string
	Service string
	Banner  string
}

func scanPort(ip string, port int, timeout time.Duration) ScanResult {
	target := fmt.Sprintf("%s:%d", ip, port)
	conn, err := net.DialTimeout("tcp", target, timeout)
	
	result := ScanResult{Port: port}
	
	if err != nil {
		result.State = "closed"
		return result
	}
	
	defer conn.Close()
	result.State = "open"
	
	// Intentar identificar el servicio
	if service, exists := commonPorts[port]; exists {
		result.Service = service
	} else {
		result.Service = "unknown"
	}
	
	// Intentar obtener un banner
	if conn != nil {
		_ = conn.SetReadDeadline(time.Now().Add(1 * time.Second))
		banner := make([]byte, 1024)
		_, err := conn.Read(banner)
		if err == nil {
			result.Banner = strings.TrimSpace(string(banner))
			if len(result.Banner) > 100 {
				result.Banner = result.Banner[:100] + "..."
			}
		}
	}
	
	return result
}

func main() {
	ip := "192.168.1.1"
	fmt.Printf("Iniciando escaneo de puertos en %s\n", ip)
	fmt.Println("---------------------------------------")
	
	// Lista de puertos a escanear
	portsToScan := []int{}
	
	// Agregar los puertos comunes
	for port := range commonPorts {
		portsToScan = append(portsToScan, port)
	}
	
	// Agregar algunos puertos adicionales comunes
	additionalPorts := []int{8000, 8008, 8081, 8888, 9000, 9090}
	portsToScan = append(portsToScan, additionalPorts...)
	
	// Ordenar puertos para una salida más legible
	sort.Ints(portsToScan)
	
	// Configurar concurrencia
	var wg sync.WaitGroup
	var mutex sync.Mutex
	results := make(map[int]ScanResult)
	timeout := 500 * time.Millisecond
	
	// Limitar la concurrencia para evitar problemas
	semaphore := make(chan struct{}, 100)
	
	for _, port := range portsToScan {
		wg.Add(1)
		semaphore <- struct{}{}
		
		go func(p int) {
			defer wg.Done()
			defer func() { <-semaphore }()
			
			result := scanPort(ip, p, timeout)
			
			mutex.Lock()
			results[p] = result
			mutex.Unlock()
		}(port)
	}
	
	wg.Wait()
	
	// Mostrar resultados
	fmt.Println("PUERTO\tESTADO\tSERVICIO\tBANNER")
	fmt.Println("---------------------------------------")
	
	openPorts := 0
	for _, port := range portsToScan {
		if result, exists := results[port]; exists && result.State == "open" {
			banner := result.Banner
			if banner != "" {
				banner = ": " + banner
			}
			fmt.Printf("%d\t%s\t%s\t%s\n", result.Port, result.State, result.Service, banner)
			openPorts++
		}
	}
	
	fmt.Println("---------------------------------------")
	fmt.Printf("Escaneo completado: %d puertos abiertos encontrados en %s\n", openPorts, ip)
}
