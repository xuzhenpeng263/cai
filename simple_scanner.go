package main

import (
	"fmt"
	"net"
	"sort"
	"sync"
	"time"
)

// Mapeo básico de puertos a servicios comunes
var commonPorts = map[int]string{
	20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet",
	25: "SMTP", 53: "DNS", 80: "HTTP", 110: "POP3",
	143: "IMAP", 443: "HTTPS", 445: "SMB", 3306: "MySQL", 
	3389: "RDP", 8080: "HTTP-Proxy", 8443: "HTTPS-Alt",
}

// Estructura para almacenar resultados
type ScanResult struct {
	Port    int
	State   string
	Service string
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
	
	// Identificar el servicio
	if service, exists := commonPorts[port]; exists {
		result.Service = service
	} else {
		result.Service = "unknown"
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
	
	// Agregar algunos puertos adicionales
	additionalPorts := []int{8000, 8008, 8081, 8888, 9000, 9090}
	portsToScan = append(portsToScan, additionalPorts...)
	
	// Ordenar puertos
	sort.Ints(portsToScan)
	
	// Configurar concurrencia
	var wg sync.WaitGroup
	var mutex sync.Mutex
	results := make(map[int]ScanResult)
	timeout := 500 * time.Millisecond
	
	// Limitar la concurrencia
	semaphore := make(chan struct{}, 50)
	
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
	fmt.Println("PUERTO\tESTADO\tSERVICIO")
	fmt.Println("---------------------------------------")
	
	openPorts := 0
	for _, port := range portsToScan {
		if result, exists := results[port]; exists && result.State == "open" {
			fmt.Printf("%d\t%s\t%s\n", result.Port, result.State, result.Service)
			openPorts++
		}
	}
	
	fmt.Println("---------------------------------------")
	fmt.Printf("Escaneo completado: %d puertos abiertos encontrados en %s\n", openPorts, ip)
}
